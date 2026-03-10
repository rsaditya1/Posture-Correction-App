# Dataset — Posture Landmark Data

## What is this?

This folder contains pose landmark data collected from a laptop webcam using MediaPipe's Pose Landmarker model. Each row represents one video frame with extracted upper-body joint angles and positional features.

## How it was collected

I sat in front of my laptop webcam and recorded myself in two types of sessions:

- **Good posture (label=1):** Sitting upright, shoulders back, head level. I moved naturally — shifted weight, turned my head slightly, etc. — so the data isn't artificially rigid.
- **Bad posture (label=0):** Slouching, hunching shoulders, leaning forward, dropping my head, turtle-necking. I varied between different bad posture types every 15-20 seconds to give the model a range of examples.

Each session was ~2 minutes at 15 frames per second.

## Features (13 columns + label)

| Feature | What it captures |
|---|---|
| `neck_incl_L` | Left ear to left shoulder angle vs vertical |
| `neck_incl_R` | Right ear to right shoulder angle vs vertical |
| `neck_incl_avg` | Average of left and right neck inclination |
| `head_forward_z` | How far nose is in front of shoulders (z-axis) |
| `nose_above_shoulder` | Vertical distance from nose to shoulder midpoint |
| `shoulder_y_diff` | Height difference between left and right shoulder |
| `shoulder_width` | Horizontal distance between shoulders |
| `ear_y_diff` | Height difference between left and right ear |
| `ear_shoulder_ratio_L` | Left ear-shoulder distance normalized by shoulder width |
| `ear_shoulder_ratio_R` | Right ear-shoulder distance normalized by shoulder width |
| `head_droop_L` | Angle at left ear between nose and left shoulder |
| `head_droop_R` | Angle at right ear between nose and right shoulder |
| `eye_ear_y_diff` | Vertical offset of eyes relative to ears (head tilt) |
| `label` | 1 = good 