import numpy as np


def compute_distance(box, boxes):
    """
    Compute spatial-temporal distance between boxes.
    Distance accounts for spatial (x, y, z) and temporal (t) components.
    """
    Xtarget = boxes[:, 3]  # x coordinate
    Ytarget = boxes[:, 2]  # y coordinate
    Ztarget = boxes[:, 1]  # z coordinate
    Ttarget = boxes[:, 0]  # time coordinate

    Xsource = box[3]
    Ysource = box[2]
    Zsource = box[1]
    Tsource = box[0]

    # Spatial-temporal distance
    # If separated in time, reduce distance to avoid multi-counting
    distance = (
        (Xtarget - Xsource) ** 2
        + (Ytarget - Ysource) ** 2
        + (Ztarget - Zsource) ** 2
        - (Ttarget - Tsource) ** 2
    )

    distance[distance < 0] = 0

    return np.sqrt(distance)


def nms_space_time(detections, nms_space=10, nms_time=2):
    """
    Non-Maximum Suppression in space and time for ONEAT detections.

    Args:
        detections: List of detection dicts with keys: time, z, y, x, predicted_class, event_name
        nms_space: Spatial distance threshold for NMS
        nms_time: Temporal distance threshold for NMS

    Returns:
        List of filtered detections after NMS
    """
    if len(detections) == 0:
        return []

    # Convert detections to numpy array for NMS
    # Format: [time, z, y, x, predicted_class]
    boxes = np.array([
        [d['time'], d['z'], d['y'], d['x'], d['predicted_class']]
        for d in detections
    ], dtype=np.float32)

    # Compute combined threshold (space + time)
    threshold = np.sqrt(nms_space ** 2 + nms_time ** 2)

    # Sort by predicted class (to prioritize certain events if needed)
    # For now, we just sort by time to process chronologically
    idxs = np.argsort(boxes[:, 0])

    pick = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        # Compute distance to remaining boxes
        if len(idxs) > 1:
            distance = compute_distance(boxes[i], boxes[idxs[1:]])

            # Remove boxes that are too close in space-time
            remove_idxs = np.where(distance < threshold)[0] + 1
            idxs = np.delete(idxs, remove_idxs)

        idxs = np.delete(idxs, 0)

    # Return filtered detections
    return [detections[i] for i in pick]


def group_detections_by_event(detections):
    """
    Group detections by event type.

    Args:
        detections: List of detection dicts

    Returns:
        Dict mapping event_name to list of detections
    """
    grouped = {}
    for det in detections:
        event_name = det['event_name']
        if event_name not in grouped:
            grouped[event_name] = []
        grouped[event_name].append(det)

    return grouped
