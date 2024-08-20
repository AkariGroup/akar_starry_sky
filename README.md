# akari_yolo_starry_sky

AKARIでYOLOの物体認識を使い、人の軌道を星空として描画するアプリです。  
夜空の背景画像上に、今認識している人の軌道を星としてプロットしつつ、過去認識した人の軌道もログから星として再現し、プロットしていきます。  
認識する人が増えるほど、星として描画される軌道も増えていきます。  
物体認識モデルとして、下記を使用しており、認識物体をpersonに制限しています。  
https://github.com/AkariGroup/akari_yolo_models/tree/main/human_parts  

## セットアップ
1. (初回のみ)submoduleの更新  
`git submodule update --init --recursive`  

1. (初回のみ)仮想環境の作成  
`python -m venv venv`  
`source venv/bin/activate`  
`pip install -r requirements.txt`  

## 実行方法  
`source venv/bin/activate`  
を実施後、下記を実行。  

`python3 starry_sky.py`  

引数は下記を指定可能  
- `-f`, `--fps`: カメラ画像の取得PFS。デフォルトは7。OAK-Dの性質上、推論の処理速度を上回る入力を与えるとアプリが異常終了しやすくなるため注意。  
- `-d`, `--display_camera`: この引数をつけると、RGB,depthの入力画像も表示される。  
- `-r`, `--robot_coordinate`: この引数をつけると、3次元位置をカメラからの相対位置でなく、ロボットからの位置に変更。AKARI本体のヘッドの向きを取得して、座標変換を行うため、ヘッドの向きによらずAKARIの正面方向の位置が表示される。  
- `--log_path`: 軌道ログを保存するディレクトリを指定。デフォルトは`./log`。また、既存のログファイルがある場合、そのファイルのパスを指定すると、そのログファイルの続きから記録、再生が可能。  
