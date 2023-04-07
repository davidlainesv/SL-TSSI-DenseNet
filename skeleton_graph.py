import re

from enum import IntEnum
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.hands import HandLandmark


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def prefix(prefix, ls):
    '''
    prefix(ls) prefixs all the elements in a list
    '''
    return [prefix + "_" + str(v) for v in ls]


class FaceLandmark(IntEnum):
    LEFT_LOWER_EYEBROW_INNER = 285
    LEFT_LOWER_EYEBROW_INNER_SECOND = 295
    LEFT_LOWER_EYEBROW_OUTER = 276
    LEFT_UPPER_EYEBROW_INNER = 336
    LEFT_UPPER_EYEBROW_OUTER = 300
    RIGHT_LOWER_EYEBROW_INNER = 55
    RIGHT_LOWER_EYEBROW_INNER_SECOND = 65
    RIGHT_LOWER_EYEBROW_OUTER = 46
    RIGHT_UPPER_EYEBROW_INNER = 107
    RIGHT_UPPER_EYEBROW_OUTER = 70
    LEFT_EYE_INNER = 362
    LEFT_EYE_INNER_SECOND = 382
    RIGHT_EYE_INNER = 133
    RIGHT_EYE_INNER_SECOND = 155
    LIPS_OUTTER = 0
    LIPS_INNER = 13


class Graph:
    def __init__(self, nodes, edges=[]):
        self.nodes = nodes.copy()
        self.num_nodes = len(nodes)
        self.edges = edges.copy()
        self.num_edges = len(edges)
        self.adj = [[0 for _ in range(self.num_nodes)]
                    for _ in range(self.num_nodes)]
        for a, b in edges:
            a_idx = self.nodes.index(a)
            b_idx = self.nodes.index(b)
            self.adj[a_idx][b_idx] = 1
            self.adj[b_idx][a_idx] = 1

    # Function to add a bidirectional edge to the graph
    def add_edge(self, start, end):
        a_idx = self.nodes.index(start)
        b_idx = self.nodes.index(end)
        self.adj[a_idx][b_idx] = 1
        self.adj[b_idx][a_idx] = 1
        self.edges.append((start, end))

    # Function to remove a bidirectional edge to the graph
    def remove_edge(self, start, end):
        a_idx = self.nodes.index(start)
        b_idx = self.nodes.index(end)
        self.adj[a_idx][b_idx] = 0
        self.adj[b_idx][a_idx] = 0
        idx = self.edges.index((start, end))
        del self.edges[idx]

    # Function to add a bidirectional edge to the graph by index
    def add_edge_by_index(self, a_idx, b_idx):
        a = self.nodes[a_idx]
        b = self.nodes[b_idx]
        self.adj[a_idx][b_idx] = 1
        self.adj[b_idx][a_idx] = 1
        self.edges.append((a, b))

    # Function to remove a bidirectional edge to the graph by index
    def remove_edge_by_index(self, a_idx, b_idx):
        a = self.nodes[a_idx]
        b = self.nodes[b_idx]
        self.adj[a_idx][b_idx] = 0
        self.adj[b_idx][a_idx] = 0
        idx = self.edges.index((a, b))
        del self.edges[idx]

    # Function to visit a node
    def visit_by_index(self, start_idx, visited):
        # Init path with the start node
        path = [start_idx]

        # Set current node as visited
        visited[start_idx] = True

        # For every node of the graph
        for i in range(self.num_nodes):
            if self.adj[start_idx][i] == 1 and not visited[i]:
                path = path + self.visit_by_index(i, visited) + [start_idx]

        return path

    # Function to perform DFS on the graph
    # Returns a list of nodes' indexes
    def dfs_by_index(self, start_idx):
        paths = []
        visited = [False] * self.num_nodes

        while True:
            paths.append(self.visit_by_index(start_idx, visited))
            if False in visited:
                start_idx = visited.index(False)
            else:
                break

        return paths


def tssi_legacy(debug=False):
    RIGHT_EYEBROW_JOINTS = [46, 52, 53, 55, 65]
    LEFT_EYEBROW_JOINTS = [276, 282, 283, 285, 295]
    LIPS_INNER_JOINTS = [13, 14, 78, 80, 81, 82, 87, 88,
                         95, 178, 191, 308, 310, 311, 312, 317, 318, 324, 402, 415]
    POSE_FACE_JOINTS = [
        PoseLandmark.NOSE.value,
        PoseLandmark.RIGHT_EYE_OUTER.value,
        PoseLandmark.RIGHT_EYE.value,
        PoseLandmark.RIGHT_EYE_INNER.value,
        PoseLandmark.LEFT_EYE_OUTER.value,
        PoseLandmark.LEFT_EYE.value,
        PoseLandmark.LEFT_EYE_INNER.value
    ]
    POSE_BODY_JOINTS = [
        PoseLandmark.RIGHT_SHOULDER.value,
        PoseLandmark.RIGHT_ELBOW.value,
        # PoseLandmark.RIGHT_WRIST.value,
        PoseLandmark.RIGHT_HIP.value,

        PoseLandmark.LEFT_SHOULDER.value,
        PoseLandmark.LEFT_ELBOW.value,
        # PoseLandmark.LEFT_WRIST.value,
        PoseLandmark.LEFT_HIP.value
    ]
    HAND_JOINTS = [
        HandLandmark.WRIST.value,
        HandLandmark.THUMB_CMC.value,
        HandLandmark.THUMB_MCP.value,
        HandLandmark.THUMB_IP.value,
        HandLandmark.THUMB_TIP.value,
        HandLandmark.INDEX_FINGER_MCP.value,
        HandLandmark.INDEX_FINGER_PIP.value,
        HandLandmark.INDEX_FINGER_DIP.value,
        HandLandmark.INDEX_FINGER_TIP.value,
        HandLandmark.MIDDLE_FINGER_MCP.value,
        HandLandmark.MIDDLE_FINGER_PIP.value,
        HandLandmark.MIDDLE_FINGER_DIP.value,
        HandLandmark.MIDDLE_FINGER_TIP.value,
        HandLandmark.RING_FINGER_MCP.value,
        HandLandmark.RING_FINGER_PIP.value,
        HandLandmark.RING_FINGER_DIP.value,
        HandLandmark.RING_FINGER_TIP.value,
        HandLandmark.PINKY_MCP.value,
        HandLandmark.PINKY_PIP.value,
        HandLandmark.PINKY_DIP.value,
        HandLandmark.PINKY_TIP.value
    ]

    FACE_JOINTS = RIGHT_EYEBROW_JOINTS + LEFT_EYEBROW_JOINTS + LIPS_INNER_JOINTS
    POSE_JOINTS = POSE_FACE_JOINTS + POSE_BODY_JOINTS

    FILTERED_FACEMESH_CONNECTIONS = [(u, v) for (
        u, v) in FACEMESH_CONTOURS if u in FACE_JOINTS and v in FACE_JOINTS]
    FILTERED_POSE_CONNECTIONS = [(u, v) for (
        u, v) in POSE_CONNECTIONS if u in POSE_JOINTS and v in POSE_JOINTS]

    joints = ['root'] + prefix('face', RIGHT_EYEBROW_JOINTS) + prefix('face', LEFT_EYEBROW_JOINTS) + prefix('pose', POSE_FACE_JOINTS) + prefix(
        'face', LIPS_INNER_JOINTS) + prefix('pose', POSE_BODY_JOINTS) + prefix('rightHand', HAND_JOINTS) + prefix('leftHand', HAND_JOINTS) + ['midhip']

    # Define graph
    graph = Graph(joints)

    # Setup connections
    for connection in FILTERED_FACEMESH_CONNECTIONS:
        start_id, end_id = connection
        start = "face_" + str(start_id)
        end = "face_" + str(end_id)
        graph.add_edge(start, end)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "leftHand_" + str(start_id)
        end = "leftHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "rightHand_" + str(start_id)
        end = "rightHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in FILTERED_POSE_CONNECTIONS:
        start_id, end_id = connection
        start = "pose_" + str(start_id)
        end = "pose_" + str(end_id)
        graph.add_edge(start, end)

    # join the nose with the left eyebrow
    graph.add_edge(
        "pose_" + str(PoseLandmark.NOSE.value),
        "face_" + str(FaceLandmark.LEFT_LOWER_EYEBROW_INNER.value))

    # join the nose with the right eyebrow
    graph.add_edge(
        "pose_" + str(PoseLandmark.NOSE.value),
        "face_" + str(FaceLandmark.RIGHT_LOWER_EYEBROW_INNER.value))

    # join the nose with the inner shape of the lips
    graph.add_edge(
        "pose_" + str(PoseLandmark.NOSE.value),
        "face_" + str(FaceLandmark.LIPS_INNER.value))

    # join the pose left wrist to the left wrist of the hand
    # graph.add_edge(
    #     "pose_" + str(PoseLandmark.LEFT_WRIST.value),
    #     "leftHand_" + str(HandLandmark.WRIST.value))

    # join the pose right wrist to the right wrist of the hand
    # graph.add_edge(
    #     "pose_" + str(PoseLandmark.RIGHT_WRIST.value),
    #     "rightHand_" + str(HandLandmark.WRIST.value))

    # join the pose left elbow to the left wrist of the hand
    graph.add_edge(
        "pose_" + str(PoseLandmark.LEFT_ELBOW.value),
        "leftHand_" + str(HandLandmark.WRIST.value))

    # join the pose right elbow to the right wrist of the hand
    graph.add_edge(
        "pose_" + str(PoseLandmark.RIGHT_ELBOW.value),
        "rightHand_" + str(HandLandmark.WRIST.value))

    # join the ROOT with the nose
    graph.add_edge(
        "root",
        "pose_" + str(PoseLandmark.NOSE.value))

    # join the ROOT with the left shoulder
    graph.add_edge(
        "root",
        "pose_" + str(PoseLandmark.LEFT_SHOULDER.value))

    # join the ROOT with the right shoulder
    graph.add_edge(
        "root",
        "pose_" + str(PoseLandmark.RIGHT_SHOULDER.value))

    # REMOVE the connection between the left shoulder and the right shoulder
    graph.remove_edge(
        "pose_" + str(PoseLandmark.LEFT_SHOULDER.value),
        "pose_" + str(PoseLandmark.RIGHT_SHOULDER.value))

    # REMOVE the connection between the left shoulder and the left hip
    graph.remove_edge(
        "pose_" + str(PoseLandmark.LEFT_SHOULDER.value),
        "pose_" + str(PoseLandmark.LEFT_HIP.value))

    # REMOVE the connection between the right shoulder and the right hip
    graph.remove_edge(
        "pose_" + str(PoseLandmark.RIGHT_SHOULDER.value),
        "pose_" + str(PoseLandmark.RIGHT_HIP.value))

    # REMOVE the connection between the left hip and the right hip
    graph.remove_edge(
        "pose_" + str(PoseLandmark.LEFT_HIP.value),
        "pose_" + str(PoseLandmark.RIGHT_HIP.value))

    # ADD the connection between the root and the midhip
    graph.add_edge(
        "root",
        "midhip")

    # ADD the connection between the root and the left hip
    graph.add_edge(
        "midhip",
        "pose_" + str(PoseLandmark.LEFT_HIP.value))

    # ADD the connection between the root and the right hip
    graph.add_edge(
        "midhip",
        "pose_" + str(PoseLandmark.RIGHT_HIP.value))

    # Perform DFS starting at the root
    root_index = graph.nodes.index("root")
    paths = graph.dfs_by_index(root_index)
    tree_path = [graph.nodes[i] for path in paths[:1] for i in path]

    # Debug info
    info = [
        ("ROOT:", tree_path.index("root")),
        ("NOSE:", tree_path.index("pose_0")),
        ("RIGHT EYEBROW:", tree_path.index("face_" +
         str(FaceLandmark.RIGHT_LOWER_EYEBROW_INNER.value))),
        ("LEFT EYEBROW:", tree_path.index("face_" +
         str(FaceLandmark.LEFT_LOWER_EYEBROW_INNER.value))),
        ("RIGHT EYE INNER:", tree_path.index(
            "pose_" + str(PoseLandmark.RIGHT_EYE_INNER.value))),
        ("LEFT EYE INNER:", tree_path.index(
            "pose_" + str(PoseLandmark.LEFT_EYE_INNER.value))),
        ("MOUTH (INNER LIPS):", tree_path.index(
            "face_" + str(FaceLandmark.LIPS_INNER.value))),
        ("RIGHT SHOULDER:", tree_path.index(
            "pose_" + str(PoseLandmark.RIGHT_SHOULDER.value))),
        ("RIGHT WRIST:", tree_path.index(
            "rightHand_" + str(HandLandmark.WRIST.value))),
        ("LEFT SHOULDER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.LEFT_SHOULDER)))),
        ("LEFT WRIST:", tree_path.index(
            "leftHand_" + str(HandLandmark.WRIST.value))),
    ]
    info.sort(key=lambda x: x[1])

    if debug:
        print(info)

    return graph, tree_path


def tssi_v2(debug=False):
    RIGHT_EYEBROW_JOINTS = [46, 52, 53, 65]
    LEFT_EYEBROW_JOINTS = [276, 282, 283, 295]
    RIGHT_EYE = [7, 159, 155, 145]
    LEFT_EYE = [382, 386, 249, 374]
    MOUTH = [324, 13, 78, 14]
    BODY_JOINTS = [
        PoseLandmark.RIGHT_SHOULDER.value,
        PoseLandmark.RIGHT_ELBOW.value,
        PoseLandmark.LEFT_SHOULDER.value,
        PoseLandmark.LEFT_ELBOW.value
    ]
    HAND_JOINTS = [
        HandLandmark.WRIST.value,
        HandLandmark.THUMB_CMC.value,
        HandLandmark.THUMB_MCP.value,
        HandLandmark.THUMB_IP.value,
        HandLandmark.THUMB_TIP.value,
        HandLandmark.INDEX_FINGER_MCP.value,
        HandLandmark.INDEX_FINGER_PIP.value,
        HandLandmark.INDEX_FINGER_DIP.value,
        HandLandmark.INDEX_FINGER_TIP.value,
        HandLandmark.MIDDLE_FINGER_MCP.value,
        HandLandmark.MIDDLE_FINGER_PIP.value,
        HandLandmark.MIDDLE_FINGER_DIP.value,
        HandLandmark.MIDDLE_FINGER_TIP.value,
        HandLandmark.RING_FINGER_MCP.value,
        HandLandmark.RING_FINGER_PIP.value,
        HandLandmark.RING_FINGER_DIP.value,
        HandLandmark.RING_FINGER_TIP.value,
        HandLandmark.PINKY_MCP.value,
        HandLandmark.PINKY_PIP.value,
        HandLandmark.PINKY_DIP.value,
        HandLandmark.PINKY_TIP.value
    ]

    FACE_JOINTS = RIGHT_EYEBROW_JOINTS + \
        LEFT_EYEBROW_JOINTS + RIGHT_EYE + LEFT_EYE + MOUTH

    FILTERED_FACEMESH_CONNECTIONS = [(u, v) for (
        u, v) in FACEMESH_CONTOURS if u in FACE_JOINTS and v in FACE_JOINTS]
    FILTERED_POSE_CONNECTIONS = [(u, v) for (
        u, v) in POSE_CONNECTIONS if u in BODY_JOINTS and v in BODY_JOINTS]

    joints = ['root', 'pose_0'] + prefix('face', FACE_JOINTS) + prefix(
        'pose', BODY_JOINTS) + prefix('rightHand', HAND_JOINTS) + prefix('leftHand', HAND_JOINTS)

    # Define graph
    graph = Graph(joints)

    # Setup connections
    for connection in FILTERED_FACEMESH_CONNECTIONS:
        start_id, end_id = connection
        start = "face_" + str(start_id)
        end = "face_" + str(end_id)
        graph.add_edge(start, end)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "leftHand_" + str(start_id)
        end = "leftHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "rightHand_" + str(start_id)
        end = "rightHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in FILTERED_POSE_CONNECTIONS:
        start_id, end_id = connection
        start = "pose_" + str(start_id)
        end = "pose_" + str(end_id)
        graph.add_edge(start, end)

    # ADD the connections in the left eye
    graph.add_edge(
        "face_" + str(155),
        "face_" + str(159))
    graph.add_edge(
        "face_" + str(159),
        "face_" + str(7))
    graph.add_edge(
        "face_" + str(7),
        "face_" + str(145))
    graph.add_edge(
        "face_" + str(145),
        "face_" + str(155))

    # ADD the connections in the right eye
    graph.add_edge(
        "face_" + str(382),
        "face_" + str(386))
    graph.add_edge(
        "face_" + str(386),
        "face_" + str(249))
    graph.add_edge(
        "face_" + str(249),
        "face_" + str(374))
    graph.add_edge(
        "face_" + str(374),
        "face_" + str(382))

    # ADD the connections in the mouth
    graph.add_edge(
        "face_" + str(13),
        "face_" + str(78))
    graph.add_edge(
        "face_" + str(78),
        "face_" + str(14))
    graph.add_edge(
        "face_" + str(14),
        "face_" + str(324))
    graph.add_edge(
        "face_" + str(324),
        "face_" + str(13))

    # ADD the connection between the nose and the left eyebrow
    graph.add_edge(
        "pose_" + str(PoseLandmark.NOSE.value),
        "face_" + str(FaceLandmark.LEFT_LOWER_EYEBROW_INNER_SECOND.value))

    # ADD the connection between the nose and the right eyebrow
    graph.add_edge(
        "pose_" + str(PoseLandmark.NOSE.value),
        "face_" + str(FaceLandmark.RIGHT_LOWER_EYEBROW_INNER_SECOND.value))

    # ADD the connection between the nose and the left eye
    graph.add_edge(
        "pose_" + str(PoseLandmark.NOSE.value),
        "face_" + str(FaceLandmark.LEFT_EYE_INNER_SECOND.value))

    # ADD the connection between the nose and the right eye
    graph.add_edge(
        "pose_" + str(PoseLandmark.NOSE.value),
        "face_" + str(FaceLandmark.RIGHT_EYE_INNER_SECOND.value))

    # ADD the connection between the nose and the mouth
    graph.add_edge(
        "pose_" + str(PoseLandmark.NOSE.value),
        "face_" + str(FaceLandmark.LIPS_INNER.value))

    # ADD the connection between the left elbow and the left wrist
    graph.add_edge(
        "pose_" + str(PoseLandmark.LEFT_ELBOW.value),
        "leftHand_" + str(HandLandmark.WRIST.value))

    # ADD the connection between the right elbow and the right wrist
    graph.add_edge(
        "pose_" + str(PoseLandmark.RIGHT_ELBOW.value),
        "rightHand_" + str(HandLandmark.WRIST.value))

    # ADD the connection between the ROOT and the nose
    graph.add_edge(
        "root",
        "pose_" + str(PoseLandmark.NOSE.value))

    # ADD the connection between the ROOT and the left shoulder
    graph.add_edge(
        "root",
        "pose_" + str(PoseLandmark.RIGHT_SHOULDER.value))

    # ADD the connection between the ROOT and the right shoulder
    graph.add_edge(
        "root",
        "pose_" + str(PoseLandmark.LEFT_SHOULDER.value))

    # REMOVE the connection between the left shoulder and the right shoulder
    graph.remove_edge(
        "pose_" + str(PoseLandmark.LEFT_SHOULDER.value),
        "pose_" + str(PoseLandmark.RIGHT_SHOULDER.value))

    # Perform DFS starting at the root
    root_index = graph.nodes.index("root")
    paths = graph.dfs_by_index(root_index)
    tree_path = [graph.nodes[i] for path in paths[:1] for i in path]

    # Debug info
    info = [
        ("ROOT:", tree_path.index("root")),
        ("NOSE:", tree_path.index("pose_0")),
        ("RIGHT EYEBROW:", tree_path.index("face_" +
         str(FaceLandmark.RIGHT_LOWER_EYEBROW_INNER_SECOND.value))),
        ("LEFT EYEBROW:", tree_path.index("face_" +
         str(FaceLandmark.LEFT_LOWER_EYEBROW_INNER_SECOND.value))),
        ("RIGHT EYE INNER:", tree_path.index(
            "face_" + str(FaceLandmark.RIGHT_EYE_INNER_SECOND.value))),
        ("LEFT EYE INNER:", tree_path.index(
            "face_" + str(FaceLandmark.LEFT_EYE_INNER_SECOND.value))),
        ("MOUTH (INNER LIPS):", tree_path.index(
            "face_" + str(FaceLandmark.LIPS_INNER.value))),
        ("RIGHT SHOULDER:", tree_path.index(
            "pose_" + str(PoseLandmark.RIGHT_SHOULDER.value))),
        ("RIGHT WRIST:", tree_path.index(
            "rightHand_" + str(HandLandmark.WRIST.value))),
        ("LEFT SHOULDER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.LEFT_SHOULDER)))),
        ("LEFT WRIST:", tree_path.index(
            "leftHand_" + str(HandLandmark.WRIST.value))),
    ]
    info.sort(key=lambda x: x[1])

    if debug:
        print(info)

    return graph, tree_path


def tssi_v3(debug=False):
    BODY_JOINTS = [
        PoseLandmark.RIGHT_SHOULDER.value,
        PoseLandmark.RIGHT_ELBOW.value,
        PoseLandmark.LEFT_SHOULDER.value,
        PoseLandmark.LEFT_ELBOW.value
    ]
    HAND_JOINTS = [
        HandLandmark.WRIST.value,
        HandLandmark.THUMB_CMC.value,
        HandLandmark.THUMB_MCP.value,
        HandLandmark.THUMB_IP.value,
        HandLandmark.THUMB_TIP.value,
        HandLandmark.INDEX_FINGER_MCP.value,
        HandLandmark.INDEX_FINGER_PIP.value,
        HandLandmark.INDEX_FINGER_DIP.value,
        HandLandmark.INDEX_FINGER_TIP.value,
        HandLandmark.MIDDLE_FINGER_MCP.value,
        HandLandmark.MIDDLE_FINGER_PIP.value,
        HandLandmark.MIDDLE_FINGER_DIP.value,
        HandLandmark.MIDDLE_FINGER_TIP.value,
        HandLandmark.RING_FINGER_MCP.value,
        HandLandmark.RING_FINGER_PIP.value,
        HandLandmark.RING_FINGER_DIP.value,
        HandLandmark.RING_FINGER_TIP.value,
        HandLandmark.PINKY_MCP.value,
        HandLandmark.PINKY_PIP.value,
        HandLandmark.PINKY_DIP.value,
        HandLandmark.PINKY_TIP.value
    ]

    FILTERED_POSE_CONNECTIONS = [(u, v) for (
        u, v) in POSE_CONNECTIONS if u in BODY_JOINTS and v in BODY_JOINTS]

    joints = ['root', 'pose_0'] + prefix(
        'pose', BODY_JOINTS) + prefix('rightHand', HAND_JOINTS) + prefix('leftHand', HAND_JOINTS)

    # Define graph
    graph = Graph(joints)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "leftHand_" + str(start_id)
        end = "leftHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "rightHand_" + str(start_id)
        end = "rightHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in FILTERED_POSE_CONNECTIONS:
        start_id, end_id = connection
        start = "pose_" + str(start_id)
        end = "pose_" + str(end_id)
        graph.add_edge(start, end)

    # ADD the connection between the left elbow and the left wrist
    graph.add_edge(
        "pose_" + str(PoseLandmark.LEFT_ELBOW.value),
        "leftHand_" + str(HandLandmark.WRIST.value))

    # ADD the connection between the right elbow and the right wrist
    graph.add_edge(
        "pose_" + str(PoseLandmark.RIGHT_ELBOW.value),
        "rightHand_" + str(HandLandmark.WRIST.value))

    # ADD the connection between the ROOT and the nose
    graph.add_edge(
        "root",
        "pose_" + str(PoseLandmark.NOSE.value))

    # ADD the connection between the ROOT and the left shoulder
    graph.add_edge(
        "root",
        "pose_" + str(PoseLandmark.RIGHT_SHOULDER.value))

    # ADD the connection between the ROOT and the right shoulder
    graph.add_edge(
        "root",
        "pose_" + str(PoseLandmark.LEFT_SHOULDER.value))

    # REMOVE the connection between the left shoulder and the right shoulder
    graph.remove_edge(
        "pose_" + str(PoseLandmark.LEFT_SHOULDER.value),
        "pose_" + str(PoseLandmark.RIGHT_SHOULDER.value))

    # Perform DFS starting at the root
    root_index = graph.nodes.index("root")
    paths = graph.dfs_by_index(root_index)
    tree_path = [graph.nodes[i] for path in paths[:1] for i in path]

    # Debug info
    info = [
        ("ROOT:", tree_path.index("root")),
        ("NOSE:", tree_path.index("pose_0")),
        ("RIGHT SHOULDER:", tree_path.index(
            "pose_" + str(PoseLandmark.RIGHT_SHOULDER.value))),
        ("RIGHT WRIST:", tree_path.index(
            "rightHand_" + str(HandLandmark.WRIST.value))),
        ("LEFT SHOULDER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.LEFT_SHOULDER)))),
        ("LEFT WRIST:", tree_path.index(
            "leftHand_" + str(HandLandmark.WRIST.value))),
    ]
    info.sort(key=lambda x: x[1])

    if debug:
        print(info)

    return graph, tree_path


def tssi_mejiaperez(debug=False):
    RIGHT_EYEBROW_JOINTS = [46, 52, 53, 65]
    LEFT_EYEBROW_JOINTS = [276, 282, 283, 295]
    RIGHT_EYE = [7, 159, 155, 145]
    LEFT_EYE = [382, 386, 249, 374]
    MOUTH = [324, 13, 78, 14]
    BODY_JOINTS = [
        PoseLandmark.RIGHT_SHOULDER.value,
        PoseLandmark.RIGHT_ELBOW.value,
        PoseLandmark.LEFT_SHOULDER.value,
        PoseLandmark.LEFT_ELBOW.value
    ]
    HAND_JOINTS = [
        HandLandmark.WRIST.value,
        HandLandmark.THUMB_CMC.value,
        HandLandmark.THUMB_MCP.value,
        HandLandmark.THUMB_IP.value,
        HandLandmark.THUMB_TIP.value,
        HandLandmark.INDEX_FINGER_MCP.value,
        HandLandmark.INDEX_FINGER_PIP.value,
        HandLandmark.INDEX_FINGER_DIP.value,
        HandLandmark.INDEX_FINGER_TIP.value,
        HandLandmark.MIDDLE_FINGER_MCP.value,
        HandLandmark.MIDDLE_FINGER_PIP.value,
        HandLandmark.MIDDLE_FINGER_DIP.value,
        HandLandmark.MIDDLE_FINGER_TIP.value,
        HandLandmark.RING_FINGER_MCP.value,
        HandLandmark.RING_FINGER_PIP.value,
        HandLandmark.RING_FINGER_DIP.value,
        HandLandmark.RING_FINGER_TIP.value,
        HandLandmark.PINKY_MCP.value,
        HandLandmark.PINKY_PIP.value,
        HandLandmark.PINKY_DIP.value,
        HandLandmark.PINKY_TIP.value
    ]

    FACE_JOINTS = RIGHT_EYEBROW_JOINTS + \
        LEFT_EYEBROW_JOINTS + RIGHT_EYE + LEFT_EYE + MOUTH

    FILTERED_FACEMESH_CONNECTIONS = [(u, v) for (
        u, v) in FACEMESH_CONTOURS if u in FACE_JOINTS and v in FACE_JOINTS]
    FILTERED_POSE_CONNECTIONS = [(u, v) for (
        u, v) in POSE_CONNECTIONS if u in BODY_JOINTS and v in BODY_JOINTS]

    joints = ['root'] + prefix('face', FACE_JOINTS) + prefix(
        'pose', BODY_JOINTS) + prefix('rightHand', HAND_JOINTS) + prefix('leftHand', HAND_JOINTS)

    # Define graph
    graph = Graph(joints)

    # Setup connections
    for connection in FILTERED_FACEMESH_CONNECTIONS:
        start_id, end_id = connection
        start = "face_" + str(start_id)
        end = "face_" + str(end_id)
        graph.add_edge(start, end)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "leftHand_" + str(start_id)
        end = "leftHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in HAND_CONNECTIONS:
        start_id, end_id = connection
        start = "rightHand_" + str(start_id)
        end = "rightHand_" + str(end_id)
        graph.add_edge(start, end)

    for connection in FILTERED_POSE_CONNECTIONS:
        start_id, end_id = connection
        start = "pose_" + str(start_id)
        end = "pose_" + str(end_id)
        graph.add_edge(start, end)

    # ADD the connections in the left eye
    graph.add_edge(
        "face_" + str(155),
        "face_" + str(159))
    graph.add_edge(
        "face_" + str(159),
        "face_" + str(7))
    graph.add_edge(
        "face_" + str(7),
        "face_" + str(145))
    graph.add_edge(
        "face_" + str(145),
        "face_" + str(155))

    # ADD the connections in the right eye
    graph.add_edge(
        "face_" + str(382),
        "face_" + str(386))
    graph.add_edge(
        "face_" + str(386),
        "face_" + str(249))
    graph.add_edge(
        "face_" + str(249),
        "face_" + str(374))
    graph.add_edge(
        "face_" + str(374),
        "face_" + str(382))

    # ADD the connections in the mouth
    graph.add_edge(
        "face_" + str(13),
        "face_" + str(78))
    graph.add_edge(
        "face_" + str(78),
        "face_" + str(14))
    graph.add_edge(
        "face_" + str(14),
        "face_" + str(324))
    graph.add_edge(
        "face_" + str(324),
        "face_" + str(13))

    # ADD the connection between the root and the left eyebrow
    graph.add_edge(
        "root",
        "face_" + str(FaceLandmark.LEFT_LOWER_EYEBROW_INNER_SECOND.value))

    # ADD the connection between the root and the right eyebrow
    graph.add_edge(
        "root",
        "face_" + str(FaceLandmark.RIGHT_LOWER_EYEBROW_INNER_SECOND.value))

    # ADD the connection between the root and the left eye
    graph.add_edge(
        "root",
        "face_" + str(FaceLandmark.LEFT_EYE_INNER_SECOND.value))

    # ADD the connection between the root and the right eye
    graph.add_edge(
        "root",
        "face_" + str(FaceLandmark.RIGHT_EYE_INNER_SECOND.value))

    # ADD the connection between the root and the mouth
    graph.add_edge(
        "root",
        "face_" + str(FaceLandmark.LIPS_INNER.value))

    # ADD the connection between the left elbow and the left wrist
    graph.add_edge(
        "pose_" + str(PoseLandmark.LEFT_ELBOW.value),
        "leftHand_" + str(HandLandmark.WRIST.value))

    # ADD the connection between the right elbow and the right wrist
    graph.add_edge(
        "pose_" + str(PoseLandmark.RIGHT_ELBOW.value),
        "rightHand_" + str(HandLandmark.WRIST.value))

    # ADD the connection between the ROOT and the left shoulder
    graph.add_edge(
        "root",
        "pose_" + str(PoseLandmark.RIGHT_SHOULDER.value))

    # ADD the connection between the ROOT and the right shoulder
    graph.add_edge(
        "root",
        "pose_" + str(PoseLandmark.LEFT_SHOULDER.value))

    # REMOVE the connection between the left shoulder and the right shoulder
    graph.remove_edge(
        "pose_" + str(PoseLandmark.LEFT_SHOULDER.value),
        "pose_" + str(PoseLandmark.RIGHT_SHOULDER.value))

    # Perform DFS starting at the root
    root_index = graph.nodes.index("root")
    paths = graph.dfs_by_index(root_index)
    tree_path = [graph.nodes[i] for path in paths[:1] for i in path]

    # Debug info
    info = [
        ("ROOT:", tree_path.index("root")),
        ("RIGHT EYEBROW:", tree_path.index("face_" +
         str(FaceLandmark.RIGHT_LOWER_EYEBROW_INNER_SECOND.value))),
        ("LEFT EYEBROW:", tree_path.index("face_" +
         str(FaceLandmark.LEFT_LOWER_EYEBROW_INNER_SECOND.value))),
        ("RIGHT EYE INNER:", tree_path.index(
            "face_" + str(FaceLandmark.RIGHT_EYE_INNER_SECOND.value))),
        ("LEFT EYE INNER:", tree_path.index(
            "face_" + str(FaceLandmark.LEFT_EYE_INNER_SECOND.value))),
        ("MOUTH (INNER LIPS):", tree_path.index(
            "face_" + str(FaceLandmark.LIPS_INNER.value))),
        ("RIGHT SHOULDER:", tree_path.index(
            "pose_" + str(PoseLandmark.RIGHT_SHOULDER.value))),
        ("RIGHT WRIST:", tree_path.index(
            "rightHand_" + str(HandLandmark.WRIST.value))),
        ("LEFT SHOULDER:", tree_path.index(
            "pose_" + str(int(PoseLandmark.LEFT_SHOULDER)))),
        ("LEFT WRIST:", tree_path.index(
            "leftHand_" + str(HandLandmark.WRIST.value))),
    ]
    info.sort(key=lambda x: x[1])

    if debug:
        print(info)

    return graph, tree_path
