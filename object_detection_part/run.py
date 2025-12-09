from ultralytics import YOLO

# 加载你的模型（你的绝对路径）python run.py

model = YOLO(r"D:\INTROAI\runs_blindroad\yv8n_merged_v4\weights\best.pt")

# 视频路径（改成你的）
video_path = r"D:\INTROAI\video\test5.mp4"

# 运行检测并保存结果
model.predict(source=video_path, save=True)

print("检测完成！带框的视频保存在 runs/detect/predict/ 下。")
