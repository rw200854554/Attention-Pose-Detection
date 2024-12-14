"# Attention-Pose-Detection" 
# Real-Time Pose Detection with YOLO

This program performs real-time pose detection using the YOLO model. It tracks user actions such as sitting, standing, and lying, along with attention-related metrics. Detection results are saved in a JSON file, allowing the analysis of average actions over time.

---

## Features

- **Real-Time Pose Detection**: Uses YOLO to detect and classify poses.
- **Attention Tracking**: Detects when the user looks away or is distracted.
- **14-Day Average Tracking**: Calculates and stores 14-day average data for user actions.
- **Customizable Inputs**: Accepts webcam or video files as input sources.
- **CLI Integration**: Command-line interface for flexibility and ease of use.

---

## Requirements

- Python 3.8+
- Libraries:
  - `ultralytics`
  - `numpy`
  - `json`
  - `os`
  - `argparse`

---

## Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```

2. Navigate to the project directory:
   ```bash
   cd <project_directory>
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run the Program
Use the command-line interface to run the detector. Below are the available arguments:

```bash
python detector.py [OPTIONS]
```

### Options

| Argument       | Default                   | Description                                     |
|----------------|---------------------------|-------------------------------------------------|
| `--model`      | `yolo11n-pose.pt`         | Path to the YOLO model file.                   |
| `--file_path`  | `records/action_data.json`| Path to the JSON file for saving pose data.    |
| `--source`     | `0`                       | Source for input (e.g., `0` for webcam, or video file path). |
| `--show`       | Disabled                  | Display detection results in a window.         |
| `--fps`        | `30`                      | Frame rate of the input source.                |

### Example Usage

#### Using Webcam:
```bash
python detector.py --source 0 --show
```

#### Using a Video File:
```bash
python detector.py --source path/to/video.mp4 --fps 60
```

#### Specifying Custom Model and Output Path:
```bash
python detector.py --model custom_model.pt --file_path custom_output.json
```

---

## Output

The program saves pose data to a JSON file specified by `--file_path`. The file structure includes:

- `sit`: Total time spent sitting.
- `stand`: Total time spent standing.
- `lying`: Total time spent lying down.
- `look_away`: Time spent looking away.
- `look_at`: Time spent paying attention after looking away.
- `distracted`: Boolean indicating whether the user is distracted.
- `14days_avg_action`: 14-day average of actions (sitting, standing, lying).
- `last_day`: Breakdown of actions from the last day.

---

## Notes

- The program creates a directory for the output file if it does not exist.
- Adjust the `fps` argument based on your input source for accurate tracking.
- The accuracy is the best when the camera's viewing direction is orthogonal to the scene and the whole body of the user is visible.


