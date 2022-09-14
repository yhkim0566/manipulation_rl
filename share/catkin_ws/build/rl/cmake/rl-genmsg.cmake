# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "rl: 0 messages, 1 services")

set(MSG_I_FLAGS "-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(rl_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/root/share/catkin_ws/src/rl/srv/SolveIk.srv" NAME_WE)
add_custom_target(_rl_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "rl" "/root/share/catkin_ws/src/rl/srv/SolveIk.srv" "geometry_msgs/Quaternion:geometry_msgs/Pose:geometry_msgs/Point:std_msgs/Float64MultiArray:std_msgs/MultiArrayLayout:std_msgs/MultiArrayDimension"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(rl
  "/root/share/catkin_ws/src/rl/srv/SolveIk.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float64MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rl
)

### Generating Module File
_generate_module_cpp(rl
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rl
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(rl_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(rl_generate_messages rl_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/root/share/catkin_ws/src/rl/srv/SolveIk.srv" NAME_WE)
add_dependencies(rl_generate_messages_cpp _rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(rl_gencpp)
add_dependencies(rl_gencpp rl_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS rl_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(rl
  "/root/share/catkin_ws/src/rl/srv/SolveIk.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float64MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/rl
)

### Generating Module File
_generate_module_eus(rl
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/rl
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(rl_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(rl_generate_messages rl_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/root/share/catkin_ws/src/rl/srv/SolveIk.srv" NAME_WE)
add_dependencies(rl_generate_messages_eus _rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(rl_geneus)
add_dependencies(rl_geneus rl_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS rl_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(rl
  "/root/share/catkin_ws/src/rl/srv/SolveIk.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float64MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rl
)

### Generating Module File
_generate_module_lisp(rl
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rl
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(rl_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(rl_generate_messages rl_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/root/share/catkin_ws/src/rl/srv/SolveIk.srv" NAME_WE)
add_dependencies(rl_generate_messages_lisp _rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(rl_genlisp)
add_dependencies(rl_genlisp rl_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS rl_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(rl
  "/root/share/catkin_ws/src/rl/srv/SolveIk.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float64MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/rl
)

### Generating Module File
_generate_module_nodejs(rl
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/rl
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(rl_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(rl_generate_messages rl_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/root/share/catkin_ws/src/rl/srv/SolveIk.srv" NAME_WE)
add_dependencies(rl_generate_messages_nodejs _rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(rl_gennodejs)
add_dependencies(rl_gennodejs rl_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS rl_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(rl
  "/root/share/catkin_ws/src/rl/srv/SolveIk.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Point.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float64MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rl
)

### Generating Module File
_generate_module_py(rl
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rl
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(rl_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(rl_generate_messages rl_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/root/share/catkin_ws/src/rl/srv/SolveIk.srv" NAME_WE)
add_dependencies(rl_generate_messages_py _rl_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(rl_genpy)
add_dependencies(rl_genpy rl_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS rl_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rl)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/rl
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(rl_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/rl)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/rl
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(rl_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rl)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/rl
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(rl_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/rl)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/rl
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(rl_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rl)
  install(CODE "execute_process(COMMAND \"/root/anaconda3/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rl\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/rl
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(rl_generate_messages_py geometry_msgs_generate_messages_py)
endif()
