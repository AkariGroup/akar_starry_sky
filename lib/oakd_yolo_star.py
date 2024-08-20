#!/usr/bin/env python

import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

from .akari_yolo_lib.oakd_tracking_yolo import (
    OakdTrackingYolo,
    PosLog,
    OrbitPlayer,
)

DISPLAY_WINDOW_SIZE_RATE = 2.0
idColors = np.random.random(size=(512, 3)) * 256
WHITE = (255, 255, 255)
STAR_COLOR = (124, 252, 244)


class OakdYoloStar(OakdTrackingYolo):
    """OAK-Dを使用してYOLO3次元物体トラッキングを行うクラス。"""

    def __init__(
        self,
        config_path: str,
        model_path: str,
        fps: int,
        fov: float = 73.0,
        cam_debug: bool = False,
        robot_coordinate: bool = False,
        track_targets: Optional[List[Union[int, str]]] = None,
        show_bird_frame: bool = True,
        show_spatial_frame: bool = False,
        show_orbit: bool = True,
        log_path: Optional[str] = "log",
    ) -> None:
        """クラスの初期化メソッド。

        Args:
            config_path (str): YOLOモデルの設定ファイルのパス。
            model_path (str): YOLOモデルファイルのパス。
            fps (int): カメラのフレームレート。
            fov (float): カメラの視野角 (degree)。defaultはOAK-D LiteのHFOVの73.0[deg]
            cam_debug (bool, optional): カメラのデバッグ用ウィンドウを表示するかどうか。デフォルトはFalse。
            robot_coordinate (bool, optional): ロボットのヘッド向きを使って物体の位置を変換するかどうか。デフォルトはFalse。
            track_targets (Optional[List[Union[int, str]]], optional): トラッキング対象のラベルリスト。デフォルトはNone。
            show_bird_frame (bool, optional): 俯瞰フレームを表示するかどうか。デフォルトはTrue。
            show_spatial_frame (bool, optional): 3次元フレームを表示するかどうか。デフォルトはFalse。
            show_orbit (bool, optional): 3次元軌道を表示するかどうか。デフォルトはFalse。
            log_path (Optional[str], optional): 物体の軌道履歴を保存するパス。show_orbitがTrueの時のみ有効。

        """
        self.BIRD_FRAME_BACKGROUND_IMAGE = (
            os.path.dirname(__file__) + "/../jpg/night_sky.jpg"
        )
        super().__init__(
            config_path=config_path,
            model_path=model_path,
            fps=fps,
            fov=fov,
            cam_debug=cam_debug,
            robot_coordinate=robot_coordinate,
            track_targets=track_targets,
            show_bird_frame=show_bird_frame,
            show_spatial_frame=show_spatial_frame,
            show_orbit=show_orbit,
            log_path=log_path,
        )
        self.max_x = 15000
        self.start_time = time.time()
        log_file_path = self.orbit_data_list.get_log_path()
        self.log_player = LogPlayer(log_file_path, start_time=self.start_time)
        self.log_player.update_bird_frame_distance(self.max_z)
        self.log_player.update_bird_frame_width(self.max_x)

    def create_bird_frame(self) -> np.ndarray:
        """
        俯瞰フレームを生成する。

        Returns:
            np.ndarray: 俯瞰フレーム。

        """
        frame = cv2.imread(str(self.BIRD_FRAME_BACKGROUND_IMAGE))
        frame = cv2.resize(frame, (960, 540))
        # cv2.rectangle(
        #    frame, (0, 283), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1
        # )
        # cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        return frame

    def update_bird_frame_width(self, distance: int) -> None:
        """俯瞰フレームの横方向の表示最大値を変更する。
        Args:
            distance (int): 最大横幅[mm]。
        """
        self.max_x = distance

    def update_bird_frame_distance(self, distance: int) -> None:
        """俯瞰フレームの距離方向の表示最大値を変更する。
        Args:
            distance (int): 最大距離[mm]。
        """
        self.max_z = distance

    def pos_to_point_x(self, frame_width: int, pos_x: float) -> int:
        """
        3次元位置をbird frame上のx座標に変換する

        Args:
            frame_width (int): bird frameの幅
            pos_x (float): 3次元位置のx

        Returns:
            int: bird frame上のx座標
        """
        return int(pos_x / self.max_x * frame_width + frame_width / 2)

    def draw_bird_frame(self, tracklets: List[Any], show_labels: bool = False) -> None:
        """
        俯瞰フレームに検出結果を描画する。

        Args:
            tracklets (List[Any]): トラッキング結果のリスト。
            show_labels (bool, optional): ラベルを表示するかどうか。デフォルトはFalse。

        """
        birds = self.bird_eye_frame.copy()
        if tracklets is not None:
            for i in range(0, len(tracklets)):
                if tracklets[i].status.name == "TRACKED":
                    point_y = self.pos_to_point_y(
                        birds.shape[0], tracklets[i].spatialCoordinates.z
                    )
                    point_x = self.pos_to_point_x(
                        birds.shape[1], tracklets[i].spatialCoordinates.x
                    )
                    if show_labels:
                        cv2.putText(
                            birds,
                            self.labels[tracklets[i].label],
                            (point_x - 30, point_y + 5),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.5,
                            (0, 255, 0),
                        )
                    cv2.circle(
                        birds,
                        (point_x, point_y),
                        2,
                        STAR_COLOR,
                        thickness=3,
                        lineType=8,
                        shift=0,
                    )
                    if self.show_orbit:
                        orbit = self.orbit_data_list.get_orbit_from_id(tracklets[i].id)
                        if orbit is not None:
                            prev_point: Optional[Tuple[int, int]] = None
                            for pos in orbit.pos_log:
                                cur_point = (
                                    self.pos_to_point_x(birds.shape[1], pos.x),
                                    self.pos_to_point_y(birds.shape[0], pos.z),
                                )
                                cv2.circle(
                                    birds,
                                    cur_point,
                                    2,
                                    STAR_COLOR,
                                    thickness=1,
                                    lineType=8,
                                    shift=0,
                                )
                                if prev_point is not None:
                                    cv2.line(
                                        birds,
                                        prev_point,
                                        cur_point,
                                        STAR_COLOR,
                                        thickness=1,
                                    )
                                prev_point = (
                                    self.pos_to_point_x(birds.shape[1], pos.x),
                                    self.pos_to_point_y(birds.shape[0], pos.z),
                                )
                            if prev_point is not None:
                                cv2.line(
                                    birds,
                                    prev_point,
                                    (point_x, point_y),
                                    STAR_COLOR,
                                    1,
                                )
        self.log_player.update_plotting_list(time.time())
        plot_logs = self.log_player.update_plot_data(time.time())
        for plot_log in plot_logs:
            point_y = self.pos_to_point_y(birds.shape[0], plot_log[1] * 1000)
            point_x = self.pos_to_point_x(birds.shape[1], plot_log[0] * 1000)
            cv2.circle(
                birds,
                (point_x, point_y),
                1,
                WHITE,
                thickness=1,
                lineType=8,
                shift=0,
            )
        cv2.imshow("birds", birds)


class LogPlayer(OrbitPlayer):
    def __init__(
        self,
        log_path: str,
        start_time: int = 0,
        duration: float = 30.0,
        speed: float = 0.03,
        fov: float = 73.0,
        max_z: float = 15000,
    ) -> None:
        """
        ログファイルを再生するクラス。

        Args:
            log_path (str): ログファイルのパス。
            start_time (int, optional): ログの再生開始時間。デフォルトは0。
            duration (float, optional): ログのplot追加の時間スケール。デフォルトは1.0。
            speed (float, optional): 再生速度。デフォルトは1.0。
            fov (float, optional): 俯瞰マップ上に描画されるOAK-Dの視野角 (度). デフォルトは 73.0 度。
            max_z (float, optional): 俯瞰マップの最大Z座標値. デフォルトは 15000。
        """
        super().__init__(log_path, speed, fov, max_z)
        self.duration = duration
        self.plotting_list: List[PosLog] = []
        self.plotting_index = 0
        self.log_path = log_path
        self.load_log(self.log_path)
        self.plot_start_time = start_time
        self.RESTART_INTERVAL = (
            10  # リセットした際に再度ログをプロットし始めるまでの時間[s]
        )
        self.last_reset_time = self.plot_start_time
        self.max_x = 15000

    def update_bird_frame_width(self, distance: int) -> None:
        """俯瞰フレームの横方向の表示最大値を変更する。
        Args:
            distance (int): 最大横幅[mm]。
        """
        self.max_x = distance

    def pos_to_point_x(self, frame_width: int, pos_x: float) -> int:
        """
        3次元位置をbird frame上のx座標に変換する

        Args:
            frame_width (int): bird frameの幅
            pos_x (float): 3次元位置のx

        Returns:
            int: bird frame上のx座標
        """
        return int(pos_x / self.max_x * frame_width + frame_width / 2)

    def load_log(self, log_path: str) -> None:
        """
        ログファイルを読み込む。

        Args:
            log_path (str): ログファイルのパス。

        """
        try:
            json_open = open(log_path, "r")
            self.log = json.load(json_open)
        except FileNotFoundError:
            print(f"Error: The file {log_path} does not exist.")
            return

    def reset_plotting_log(self, cur_time: int) -> None:
        """プロットする物体リストをリセットする。"""
        self.plotting_index = 0
        self.load_log(self.log_path)
        self.plot_start_time = cur_time
        self.last_reset_time = cur_time

    def update_plotting_list(self, cur_time: int) -> None:
        """
        指定した時間のプロットする物体リストを更新する。

        Args:
            cur_time (int): 時間。

        """
        updated_plotting_list = []
        for data in self.plotting_list:
            if self.get_plot_pos(cur_time, data) is not None:
                updated_plotting_list.append(data)
        while True:
            if self.plotting_index >= len(self.log["logs"]):
                self.reset_plotting_log(cur_time)
                break
            if (
                self.log["logs"][self.plotting_index]["time"] / self.duration
                - (cur_time - self.plot_start_time)
            ) <= 0:
                # timeを現在時刻に更新した上でplotting_listに追加
                is_available = False
                for data in updated_plotting_list:
                    if data["id"] == self.log["logs"][self.plotting_index]["id"]:
                        is_available = True
                if not is_available:
                    new_data = copy.deepcopy(self.log["logs"][self.plotting_index])
                    new_data["time"] = cur_time
                    updated_plotting_list.append(new_data)
                self.plotting_index += 1
            else:
                break
        self.plotting_list = updated_plotting_list

    def get_plot_pos(self, cur_time: int, pos_log: PosLog) -> Optional[Tuple[int, int]]:
        """
        指定した時間の位置を取得する。

        Args:
            cur_time (int): 時間。
            pos_log (PosLog): 時間と位置のログ。

        Returns:
            Tuple[int, int]: 位置。

        """
        (decimal, index) = math.modf(
            (cur_time - pos_log["time"]) / self.interval * self.speed
        )
        index = int(index)
        if index >= len(pos_log["pos"]) - 1:
            return None
        return (
            pos_log["pos"][index][0] * (1 - decimal)
            + pos_log["pos"][index + 1][0] * decimal,
            pos_log["pos"][index][2] * (1 - decimal)
            + pos_log["pos"][index + 1][2] * decimal,
        )

    def update_plot_data(self, cur_time: int) -> List[Tuple[float, float]]:
        """
        指定した時間の位置リストを取得する。

        Args:
            cur_time (int): 時間。

        Returns:
            List[Tuple[float, float]]: 位置リスト。
        """
        self.update_plotting_list(cur_time)
        pos_list = []
        if self.plotting_list is not None:
            for data in self.plotting_list:
                pos = self.get_plot_pos(cur_time, data)
                if pos is not None:
                    pos_list.append(pos)
        return pos_list
