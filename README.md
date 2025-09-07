# Accelerometer-based-Twist-Compensation-for-Klipper
Uses accelerometer readings at various points on an axis of motion and derives the twist angle along that axis. Uses the twist angle and nozzle_axis offsets to determine Z offset corrections along the axis.

**Summary of this method and why it works**

1. The accelerometer captures all motion including rotation-induced components. We are going to use the accelerometer as a gyroscope at various points on the 3D printer's motion axis.

2. Rotation around any single axis produces acceleration vectors lying roughly in a plane perpendicular to the rotation axis. If we mount the accelerometer on the toolhead, and move the toolhead in a non-accelerating fashion in only one of our 3D printer's axes, taking readings along the way, we can extract the axis of linear motion. Later and more importantly, if the axis has a twist in it, we can also extract the rotation rate along that axis.

3. [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) reliably identifies the rotation axis automatically, even if the device starts at an arbitrary orientation. PCA works purely on relative changes in the accelerometer vectors, so the initial orientation is irrelevant. We can mount the accelerometer in any orientation we like on the toolhead, using common sense of course. 

4. We then project the 3D accelerometer vectors onto the plane perpendicular to the rotation axis, thereby removing any component along the rotation axis. In this projected 2D plane, the vectors fully represent the rotation only. The change in rotation, $\delta \theta$, is consistently detected by comparing n = atan2(x<sub>n</sub>', y<sub>n</sub>') where x', y' are the projected vectors.

Continuous data capture during 20mm/s movement (low signal/noise)
<img width="1000" height="500" alt="20mms_moving" src="https://github.com/user-attachments/assets/5cf3e8e6-f662-47f6-8da1-b9be65224f0e" />
