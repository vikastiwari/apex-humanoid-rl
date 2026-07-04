# Simulation Rendering Design (Adapted from UI_UX_DESIGN)

> *Note: This document replaces the standard UI_UX_DESIGN, focusing on visual debugging and MuJoCo simulation rendering.*

## Passive Viewer Rendering
Training happens completely headlessly in GPU memory (`mjx`). However, we need visual confirmation of the policies.
We use the standard `mujoco.viewer` to render policies running on the CPU.

### Best Practices for Visual Debugging
- **Camera Tracking:** When writing custom rendering logic, bind the camera to the center of mass of the H1 robot (the `torso` body). Otherwise, the robot will quickly run out of frame.
- **Contact Forces:** Enable contact force rendering in the MuJoCo viewer settings to visualize ground reaction forces (GRFs). This is critical for identifying whether the robot is balancing correctly or simply exploiting soft collisions.

## Headless Video Capture
For presentation (e.g. portfolio demonstrations), we use `mujoco.Renderer` offscreen capabilities to capture frames and compile them into MP4 files at 60 FPS. Ensure the lighting parameters in `h1.xml` are tuned for contrast so the wireframe collisions are visible.
