import subprocess


for task in ["aspect_category", "aspect_category_sentiment", "end_2_end_absa", "target_aspect_sentiment_detection"]:
    for model_type in ["base", "large"]:
        command = f"accelerate launch --multi_gpu --num_processes=2 train_baseline.py {task} {model_type}"
        process = subprocess.Popen(command, shell=True)
        process.wait()
