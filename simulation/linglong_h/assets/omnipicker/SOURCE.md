# AgiBot OmniPicker assets

These ten convex STL files are copied without geometric modification from the
official `AgibotTech/genie_sim` repository, path:

`source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/robots/genie/g2/meshes/gripper/omnipicker/convex/`

Source: https://github.com/AgibotTech/genie_sim

The accompanying `../../omnipicker.xml` transcribes the official OmniPicker
link transforms, joint limits, mimic ratios, masses, and inertias from
`xacro/gripper.omnipicker.urdf.xacro`.

`ur5e_adapter.stl` is triangulated from the user-provided
`/home/lenovo/Downloads/连接件.STEP`. Its native STEP bounds are
63.0 x 63.0 x 16.0 mm and the MJCF applies a 0.001 millimetre-to-metre scale.
Its mounting axis is taken from STEP +Z; the final rotation about that axis
must still be checked against the physical UR5e/OmniPicker assembly.

The G2-specific `camera_link` is intentionally omitted from the UR5e assembly;
the user's hardware setup uses the supplied adapter and OmniPicker only.
