# Algorithm Design - Vision Module

## 4.1 Algorithm Design

Mention the algorithm(s) used in your project to get the work done with regards to major modules. Provide a pseudocode explanation regarding the functioning of the core features. Following are few examples of algorithms/pseudocode.

---

## Example:

### Algorithm 1: Fall Detection Based on YOLOv8 Object Detection

**Input:** Video frame F, Pre-trained YOLOv8 model M, Confidence threshold τ = 0.5  
**Output:** Detection result D = {class, confidence, bounding_box}

```
1.  Load pre-trained YOLOv8 model M
2.  Preprocess frame F:
    F_resized ← resize(F, 640×480)
    F_normalized ← normalize(F_resized)
3.  Run inference:
    predictions ← M.predict(F_normalized, conf=τ)
4.  Extract detections:
    for each detection in predictions do:
        class_id ← detection.class_id
        confidence ← detection.confidence
        bbox ← detection.bounding_box
        if (class_id == 1) then
            D.class ← "Fall"
        else if (class_id == 0) then
            D.class ← "Normal"
        end if
        D.confidence ← confidence
        D.bounding_box ← bbox
    end for
5.  Filter detections:
    if (D.confidence ≥ τ) then
        return D
    else
        return null
    end if
6.  Post-process:
    if (D.class == "Fall") then
        trigger_alert("Fall detected", D.confidence)
    end if
7.  return D
```

---

### Algorithm 2: Seizure Detection Based on YOLOv8 Classification

**Input:** Image frame I, Pre-trained YOLOv8 model M_seizure, Confidence threshold τ = 0.25  
**Output:** Classification result C = {class, confidence, prediction_status}

```
1.  Load pre-trained seizure detection model M_seizure
2.  Preprocess image I:
    I_resized ← resize(I, 640×480)
    I_normalized ← normalize(I_resized)
3.  Run model inference:
    results ← M_seizure.predict(I_normalized, conf=τ)
4.  Extract predictions:
    if (results.boxes is not empty) then
        for each box in results.boxes do:
            class_id ← int(box.class_id)
            confidence ← float(box.confidence)
            if (class_id == 1) then
                C.class ← "Seizure"
            else if (class_id == 0) then
                C.class ← "Normal"
            end if
            C.confidence ← confidence
        end for
    else
        C.class ← "None"
        C.confidence ← 0.0
    end if
5.  Validate prediction:
    if (C.confidence ≥ τ) then
        C.prediction_status ← "Valid"
    else
        C.prediction_status ← "Low_Confidence"
    end if
6.  Decision making:
    if (C.class == "Seizure" AND C.prediction_status == "Valid") then
        trigger_alert("Seizure detected", C.confidence)
        log_event(C)
    end if
7.  return C
```

---

### Algorithm 3: Vision Module Frame Processing Pipeline

**Input:** Video stream S, Fall model M_fall, Seizure model M_seizure  
**Output:** Event list E = {fall_events, seizure_events}

```
1.  Initialize:
    E.fall_events ← []
    E.seizure_events ← []
    frame_buffer ← deque(maxlen=30)
2.  For each frame F in video stream S do:
    a.  Preprocess frame:
        F_processed ← preprocess(F)
        frame_buffer.append(F_processed)
    b.  Fall Detection:
        fall_result ← Algorithm1(F_processed, M_fall, τ_fall=0.5)
        if (fall_result ≠ null AND fall_result.class == "Fall") then
            event ← {
                type: "fall",
                confidence: fall_result.confidence,
                timestamp: current_time(),
                bbox: fall_result.bounding_box
            }
            E.fall_events.append(event)
        end if
    c.  Seizure Detection:
        seizure_result ← Algorithm2(F_processed, M_seizure, τ_seizure=0.25)
        if (seizure_result.class == "Seizure" AND 
            seizure_result.prediction_status == "Valid") then
            event ← {
                type: "seizure",
                confidence: seizure_result.confidence,
                timestamp: current_time()
            }
            E.seizure_events.append(event)
        end if
    d.  Temporal smoothing (optional):
        if (len(frame_buffer) ≥ 5) then
            recent_falls ← count_falls_in_buffer(frame_buffer)
            if (recent_falls ≥ 3) then
                confirm_fall_event()
            end if
        end if
    end for
3.  Aggregate results:
    total_events ← len(E.fall_events) + len(E.seizure_events)
4.  return E
```

---

## Algorithm Details

### Fall Detection Algorithm (Algorithm 1)
- **Model:** YOLOv8n (nano variant for real-time performance)
- **Classes:** 0 = Normal, 1 = Fall
- **Confidence Threshold:** 0.5 (50%)
- **Input Resolution:** 640×480 pixels
- **Output:** Bounding box coordinates, class label, confidence score
- **Post-processing:** Alert generation when fall is detected with confidence ≥ threshold

### Seizure Detection Algorithm (Algorithm 2)
- **Model:** YOLOv8 (custom trained on seizure dataset)
- **Classes:** 0 = Normal, 1 = Seizure
- **Confidence Threshold:** 0.25 (25% - lower for better recall)
- **Input Resolution:** 640×480 pixels
- **Output:** Class label, confidence score, prediction status
- **Post-processing:** Alert generation and event logging when seizure detected

### Frame Processing Pipeline (Algorithm 3)
- **Purpose:** Integrates both detection modules for real-time video processing
- **Frame Buffer:** Maintains last 30 frames for temporal analysis
- **Temporal Smoothing:** Reduces false positives by requiring multiple detections
- **Event Aggregation:** Combines results from both modules into unified event list

---

## Key Features

1. **Real-time Processing:** Both algorithms process frames at video frame rate (30 FPS)
2. **Confidence-based Filtering:** Only high-confidence detections trigger alerts
3. **Temporal Consistency:** Frame buffer enables temporal smoothing for more reliable detection
4. **Modular Design:** Each detection module operates independently and can be enabled/disabled
5. **Alert System:** Automatic alert generation when critical events are detected

