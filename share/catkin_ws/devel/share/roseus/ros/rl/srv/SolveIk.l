;; Auto-generated. Do not edit!


(when (boundp 'rl::SolveIk)
  (if (not (find-package "RL"))
    (make-package "RL"))
  (shadow 'SolveIk (find-package "RL")))
(unless (find-package "RL::SOLVEIK")
  (make-package "RL::SOLVEIK"))
(unless (find-package "RL::SOLVEIKREQUEST")
  (make-package "RL::SOLVEIKREQUEST"))
(unless (find-package "RL::SOLVEIKRESPONSE")
  (make-package "RL::SOLVEIKRESPONSE"))

(in-package "ROS")

(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass rl::SolveIkRequest
  :super ros::object
  :slots (_end_pose ))

(defmethod rl::SolveIkRequest
  (:init
   (&key
    ((:end_pose __end_pose) (instance geometry_msgs::Pose :init))
    )
   (send-super :init)
   (setq _end_pose __end_pose)
   self)
  (:end_pose
   (&rest __end_pose)
   (if (keywordp (car __end_pose))
       (send* _end_pose __end_pose)
     (progn
       (if __end_pose (setq _end_pose (car __end_pose)))
       _end_pose)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Pose _end_pose
    (send _end_pose :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Pose _end_pose
       (send _end_pose :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Pose _end_pose
     (send _end_pose :deserialize buf ptr-) (incf ptr- (send _end_pose :serialization-length))
   ;;
   self)
  )

(defclass rl::SolveIkResponse
  :super ros::object
  :slots (_ik_result _success ))

(defmethod rl::SolveIkResponse
  (:init
   (&key
    ((:ik_result __ik_result) (instance std_msgs::Float64MultiArray :init))
    ((:success __success) nil)
    )
   (send-super :init)
   (setq _ik_result __ik_result)
   (setq _success __success)
   self)
  (:ik_result
   (&rest __ik_result)
   (if (keywordp (car __ik_result))
       (send* _ik_result __ik_result)
     (progn
       (if __ik_result (setq _ik_result (car __ik_result)))
       _ik_result)))
  (:success
   (&optional (__success :null))
   (if (not (eq __success :null)) (setq _success __success)) _success)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Float64MultiArray _ik_result
    (send _ik_result :serialization-length)
    ;; bool _success
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Float64MultiArray _ik_result
       (send _ik_result :serialize s)
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Float64MultiArray _ik_result
     (send _ik_result :deserialize buf ptr-) (incf ptr- (send _ik_result :serialization-length))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass rl::SolveIk
  :super ros::object
  :slots ())

(setf (get rl::SolveIk :md5sum-) "6580254afbe9e0087fa243dc6fb4dd77")
(setf (get rl::SolveIk :datatype-) "rl/SolveIk")
(setf (get rl::SolveIk :request) rl::SolveIkRequest)
(setf (get rl::SolveIk :response) rl::SolveIkResponse)

(defmethod rl::SolveIkRequest
  (:response () (instance rl::SolveIkResponse :init)))

(setf (get rl::SolveIkRequest :md5sum-) "6580254afbe9e0087fa243dc6fb4dd77")
(setf (get rl::SolveIkRequest :datatype-) "rl/SolveIkRequest")
(setf (get rl::SolveIkRequest :definition-)
      "geometry_msgs/Pose end_pose

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
---
std_msgs/Float64MultiArray ik_result
bool success


================================================================================
MSG: std_msgs/Float64MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
float64[]         data          # array of data


================================================================================
MSG: std_msgs/MultiArrayLayout
# The multiarray declares a generic multi-dimensional array of a
# particular data type.  Dimensions are ordered from outer most
# to inner most.

MultiArrayDimension[] dim # Array of dimension properties
uint32 data_offset        # padding elements at front of data

# Accessors should ALWAYS be written in terms of dimension stride
# and specified outer-most dimension first.
# 
# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
#
# A standard, 3-channel 640x480 image with interleaved color channels
# would be specified as:
#
# dim[0].label  = \"height\"
# dim[0].size   = 480
# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
# dim[1].label  = \"width\"
# dim[1].size   = 640
# dim[1].stride = 3*640 = 1920
# dim[2].label  = \"channel\"
# dim[2].size   = 3
# dim[2].stride = 3
#
# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.

================================================================================
MSG: std_msgs/MultiArrayDimension
string label   # label of given dimension
uint32 size    # size of given dimension (in type units)
")

(setf (get rl::SolveIkResponse :md5sum-) "6580254afbe9e0087fa243dc6fb4dd77")
(setf (get rl::SolveIkResponse :datatype-) "rl/SolveIkResponse")
(setf (get rl::SolveIkResponse :definition-)
      "geometry_msgs/Pose end_pose

================================================================================
MSG: geometry_msgs/Pose
# A representation of pose in free space, composed of position and orientation. 
Point position
Quaternion orientation

================================================================================
MSG: geometry_msgs/Point
# This contains the position of a point in free space
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
---
std_msgs/Float64MultiArray ik_result
bool success


================================================================================
MSG: std_msgs/Float64MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
float64[]         data          # array of data


================================================================================
MSG: std_msgs/MultiArrayLayout
# The multiarray declares a generic multi-dimensional array of a
# particular data type.  Dimensions are ordered from outer most
# to inner most.

MultiArrayDimension[] dim # Array of dimension properties
uint32 data_offset        # padding elements at front of data

# Accessors should ALWAYS be written in terms of dimension stride
# and specified outer-most dimension first.
# 
# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
#
# A standard, 3-channel 640x480 image with interleaved color channels
# would be specified as:
#
# dim[0].label  = \"height\"
# dim[0].size   = 480
# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
# dim[1].label  = \"width\"
# dim[1].size   = 640
# dim[1].stride = 3*640 = 1920
# dim[2].label  = \"channel\"
# dim[2].size   = 3
# dim[2].stride = 3
#
# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.

================================================================================
MSG: std_msgs/MultiArrayDimension
string label   # label of given dimension
uint32 size    # size of given dimension (in type units)
")



(provide :rl/SolveIk "6580254afbe9e0087fa243dc6fb4dd77")


