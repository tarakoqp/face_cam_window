# face_cam

Web カメラ映像から顔・手のランドマークを検出し、輪郭を描画する Python アプリケーションです。  
MediaPipe と OpenCV を使用しています。

## 必要環境

- Python 3.8 以上
- pip

## セットアップ

1. リポジトリをクローン

   ```
   git clone https://github.com/あなたのユーザー名/face_cam.git
   cd face_cam
   ```

2. 仮想環境の作成・有効化（任意）

   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. 依存パッケージのインストール
   ```
   pip install -r requirements.txt
   ```

## 使い方

```
python main.py
```

- Web カメラが必要です。
- `Esc`キーで終了します。

## exe 化（Windows）

PyInstaller で exe ファイルを作成できます。

```
python -m PyInstaller --onefile --noconsole --add-data ".venv\Lib\site-packages\mediapipe\modules;mediapipe\modules" main.py
```

## ライセンス

MIT License
