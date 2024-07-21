#!/usr/bin/env python

import contextlib
import copy
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import blobconverter
import cv2
import depthai as dai
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .util import HostSync, TextHelper
from akari_yolo_lib.oakd_tracking_yolo import OakdTrackingYolo, PosLog

DISPLAY_WINDOW_SIZE_RATE = 2.0
idColors = np.random.random(size=(256, 3)) * 256


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
        self.start_time = datetime.now()
        self.log_player = LogPlayer()

    def get_frame(self) -> Union[np.ndarray, List[Any], Any]:
        """フレーム画像と検出結果を取得する。

        Returns:
            Union[np.ndarray, List[Any]]: フレーム画像と検出結果のリストのタプル。

        """
        frame = None
        detections = []
        ret = False
        try:
            ret = self.qRgb.has()
            if ret:
                rgb_mes = self.qRgb.get()
                self.sync.add_msg("rgb", rgb_mes)
                if self.robot_coordinate:
                    self.sync.add_msg(
                        "head_pos",
                        self.joints.get_joint_positions(),
                        str(rgb_mes.getSequenceNum()),
                    )
        except BaseException:
            raise
        ret = False
        try:
            ret = self.qDepth.has()
            if ret:
                self.sync.add_msg("depth", self.qDepth.get())
        except BaseException:
            raise
        ret = False
        try:
            ret = self.qRaw.has()
            if ret:
                self.sync.add_msg("raw", self.qRaw.get())
        except BaseException:
            raise
        ret = False
        try:
            ret = self.qDet.has()
            if ret:
                self.sync.add_msg("detections", self.qDet.get())
                self.counter += 1
        except BaseException:
            raise
        ret = False
        try:
            ret = self.qTrack.has()
            if ret:
                self.track = self.qTrack.get()
        except BaseException:
            raise
        msgs = self.sync.get_msgs()
        tracklets = None
        if msgs is not None:
            detections = msgs["detections"].detections
            frame = msgs["rgb"].getCvFrame()
            depthFrame = msgs["depth"].getFrame()
            self.raw_frame = msgs["raw"].getCvFrame()
            depthFrameColor = cv2.normalize(
                depthFrame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3
            )
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, self.jet_custom)
            if self.cam_debug:
                cv2.imshow("rgb", cv2.resize(frame, (self.width, self.height)))
                cv2.imshow(
                    "depth",
                    cv2.resize(depthFrameColor, (self.width, int(self.width * 3 / 4))),
                )
            height = int(frame.shape[1] * 9 / 16)
            width = frame.shape[1]
            brank_height = width - height
            frame = frame[
                int(brank_height / 2) : int(frame.shape[0] - brank_height / 2),
                0:width,
            ]
            for detection in detections:
                # Fix ymin and ymax to cropped frame pos
                detection.ymin = (width / height) * detection.ymin - (
                    brank_height / 2 / height
                )
                detection.ymax = (width / height) * detection.ymax - (
                    brank_height / 2 / height
                )
            if self.track is not None:
                tracklets = self.track.tracklets
                for tracklet in tracklets:
                    # Fix roi to cropped frame pos
                    tracklet.roi.y = (width / height) * tracklet.roi.y - (
                        brank_height / 2 / height
                    )
                    tracklet.roi.height = tracklet.roi.height * width / height

            if self.robot_coordinate:
                self.pos = msgs["head_pos"]
                for detection in detections:
                    converted_pos = self.convert_to_pos_from_akari(
                        detection.spatialCoordinates, self.pos["tilt"], self.pos["pan"]
                    )
                    detection.spatialCoordinates.x = converted_pos[0][0]
                    detection.spatialCoordinates.y = converted_pos[1][0]
                    detection.spatialCoordinates.z = converted_pos[2][0]
                for tracklet in tracklets:
                    converted_pos = self.convert_to_pos_from_akari(
                        tracklet.spatialCoordinates, self.pos["tilt"], self.pos["pan"]
                    )
                    tracklet.spatialCoordinates.x = converted_pos[0][0]
                    tracklet.spatialCoordinates.y = converted_pos[1][0]
                    tracklet.spatialCoordinates.z = converted_pos[2][0]
            if self.show_orbit:
                self.orbit_data_list.update_orbit_data(tracklets)
        return frame, detections, tracklets

    def get_labeled_frame(
        self,
        frame: np.ndarray,
        tracklets: List[Any],
        id: Optional[int] = None,
        disp_info: bool = False,
    ) -> np.ndarray:
        """認識結果をフレーム画像に描画する。

        Args:
            frame (np.ndarray): 画像フレーム。
            tracklets (List[Any]): トラッキング結果のリスト。
            id (Optional[int], optional): 描画するオブジェクトのID。指定すると、そのIDのみを描画した画像フレームを返す。指定しない場合は全てのオブジェクトを描画する。
            disp_info (bool, optional): クラス名とconfidenceをフレーム内に表示するかどうか。デフォルトはFalse。

        Returns:
            np.ndarray: 描画された画像フレーム。

        """
        for tracklet in tracklets:
            if id is not None and tracklet.id != id:
                continue
            if tracklet.status.name == "TRACKED":
                roi = tracklet.roi.denormalize(frame.shape[1], frame.shape[0])
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)
                try:
                    label = self.labels[tracklet.label]
                except Exception:
                    label = tracklet.label
                self.text.rectangle(frame, (x1, y1), (x2, y2), idColors[tracklet.id])
                if disp_info:
                    self.text.put_text(frame, str(label), (x1 + 10, y1 + 20))
                    self.text.put_text(
                        frame,
                        f"ID: {[tracklet.id]}",
                        (x1 + 10, y1 + 45),
                    )
                    self.text.put_text(frame, tracklet.status.name, (x1 + 10, y1 + 70))

                    if tracklet.spatialCoordinates.z != 0:
                        self.text.put_text(
                            frame,
                            "X: {:.2f} m".format(tracklet.spatialCoordinates.x / 1000),
                            (x1 + 10, y1 + 95),
                        )
                        self.text.put_text(
                            frame,
                            "Y: {:.2f} m".format(tracklet.spatialCoordinates.y / 1000),
                            (x1 + 10, y1 + 120),
                        )
                        self.text.put_text(
                            frame,
                            "Z: {:.2f} m".format(tracklet.spatialCoordinates.z / 1000),
                            (x1 + 10, y1 + 145),
                        )
        return frame

    def display_frame(self, name: str, frame: np.ndarray, tracklets: List[Any]) -> None:
        """画像フレームと認識結果を描画する。

        Args:
            name (str): ウィンドウ名。
            frame (np.ndarray): 画像フレーム。
            tracklets (List[Any]): トラッキング結果のリスト。
            birds(bool): 俯瞰フレームを表示するかどうか。デフォルトはTrue。

        """
        if frame is not None:
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] * DISPLAY_WINDOW_SIZE_RATE),
                    int(frame.shape[0] * DISPLAY_WINDOW_SIZE_RATE),
                ),
            )
            if tracklets is not None:
                self.get_labeled_frame(frame=frame, tracklets=tracklets, disp_info=True)
            cv2.putText(
                frame,
                "NN fps: {:.2f}".format(
                    self.counter / (time.monotonic() - self.start_time)
                ),
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.3,
                (255, 255, 255),
            )
            # Show the frame
            cv2.imshow(name, frame)
            if self.show_bird_frame:
                self.draw_bird_frame(tracklets)
            if self.show_spatial_frame:
                self.draw_spatial_frame(tracklets)

    def create_bird_frame(self) -> np.ndarray:
        """
        俯瞰フレームを生成する。

        Returns:
            np.ndarray: 俯瞰フレーム。

        """
        fov = self.fov
        frame = np.zeros((300, 300, 3), np.uint8)
        cv2.rectangle(
            frame, (0, 283), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1
        )

        alpha = (180 - fov) / 2
        center = int(frame.shape[1] / 2)
        max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
        fov_cnt = np.array(
            [
                (0, frame.shape[0]),
                (frame.shape[1], frame.shape[0]),
                (frame.shape[1], max_p),
                (center, frame.shape[0]),
                (0, max_p),
                (0, frame.shape[0]),
            ]
        )
        cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        return frame

    def update_bird_frame_distance(self, distance: int) -> None:
        """俯瞰フレームの距離方向の表示最大値を変更する。
        Args:
            distance (int): 最大距離[mm]。
        """
        self.max_z = distance

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
                        idColors[tracklets[i].id],
                        thickness=5,
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
                                    idColors[tracklets[i].id],
                                    thickness=2,
                                    lineType=8,
                                    shift=0,
                                )
                                if prev_point is not None:
                                    cv2.line(
                                        birds,
                                        prev_point,
                                        cur_point,
                                        idColors[tracklets[i].id],
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
                                    idColors[tracklets[i].id],
                                    2,
                                )

        cv2.imshow("birds", birds)


class LogPlayer(OrbitPlayer):
    def __init__(
        self,
        log_path: str,
        duration: float = 1.0,
        speed: float = 1.0,
        fov: float = 73.0,
        max_z: float = 15000,
    ) -> None:
        super().__init__(self, log_path, speed, fov, max_z)
        self.duration = duration
        self.plotting_list: List[PosLog] = []

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

    def get_plot_pos(datetime: datetime, pos_log:PosLog) -> Tuple[int, int]:
        """
        指定した時間の位置を取得する。

        Args:
            datetime (datetime): 時間。
            pos_log (PosLog): 時間と位置のログ。

        Returns:
            Tuple[int, int]: 位置。

        """
        index = (datetime - pos_log["time"]) / self.interval / self.speed

        return None

    def update_plotting_list(datetime: datetime) -> List[OrbitData]:
        """
        指定した時間の軌道データのリストを取得する。

        Args:
            datetime (datetime): 時間。

        Returns:
            List[OrbitData]: 軌道データのリスト。

        """
        orbit_data_list = []
        for orbit_data in self.log:
            if orbit_data["datetime"] == datetime:
                orbit_data_list.append(orbit_data)
        return orbit_data_list
