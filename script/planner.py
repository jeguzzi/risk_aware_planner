#!/usr/bin/env python


import itertools

import numpy as np

import networkx as nx
import rospy
import scipy.misc
import tf2_ros
from geometry_msgs.msg import PoseStamped
from matplotlib import pyplot as plt
from path_follower import array_from_msg, pose_in_frame
from traversability_rviz_paths.msg import Path, Paths
from sensor_msgs.msg import Image
from shapely.geometry import LineString
from std_msgs.msg import ColorRGBA


def node(ix, iy, cols):
    return iy * cols + ix


def nodeAtCoordinatates(coords, origin, stride, cols):
    e = stride / np.linalg.norm(stride, axis=1) ** 2
    x, y = np.round(np.dot((coords - origin), e))
    return node(int(x), int(y), cols)


def nccr_origin(g):
    return g.pos[0][:2]


def nccr_stride(g, cols):
    return (g.pos[1] - g.pos[0])[:2], (g.pos[cols] - g.pos[0])[:2]


def clamp(v, _max, top):

    return v if v < _max else top


def nccr_edge(e, max_traversability=0.99):
    x, y, data = e
    return (int(x), int(y), {
        'survival': max(0, -np.log(clamp(data['probability'], max_traversability, 1.0))),
        'distance': data['distance'],
        'p': data['probability']})


def nccr_pos(data, m=3):
    """
    extract node's map pose from original Omar's graph
     rviz_pos: 4.35,4.35,0.0 -> x: 4.35, y:4.35, z=0.0
    """
    return [round(float(t), m) for t in data['rviz_pos'].split(',')]


def nccr_graph(G, min_traversability=0, max_traversability=0.99):
    g = nx.DiGraph()
    g.add_edges_from([nccr_edge(e, max_traversability=max_traversability)
                      for e in G.edges(data=True)
                      if e[2]['probability'] > min_traversability])
    for node, data in g.nodes(data=True):
        data['x'], data['y'], data['z'] = nccr_pos(G.node[str(node)])
    g.pos = {node: np.array([data['x'], data['y'], data['z']])
             for node, data in g.nodes(data=True)}
    return g


def n_grams(l, n):
    z = (itertools.islice(l, i, None) for i in range(n))
    return zip(*z)


def simple_cull(inputPoints, dominates, mtol=1.0):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row, mtol=mtol):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow, mtol=mtol):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints


def dominates(row, candidateRow, mtol=1.0):
    return sum([row[x] <= candidateRow[x] * mtol
                for x in range(1, len(row))]) == (len(row) - 1)


def comb_path(G, k, s, t, key='distance'):
    for _, _, d in G.edges(data=True):
        d['weight'] = (1 - k) * d[key] + k * d['survival']
        if d['weight'] < 0:
            print(k, d[key], d['survival'])
    try:
        ps = list(itertools.islice(
            nx.all_shortest_paths(G, s, t, weight='weight'), 50))
    except Exception as e:
        rospy.error('No path', k, s, t)

    attrss = [[G[x][y] for x, y in n_grams(p, 2)] for p in ps]
    if len(ps) > 1:
        costs = [(i, sum(a['survival'] for a in attrs), sum(a[key] for a in attrs))
                 for i, attrs in enumerate(attrss)]
        pareto_points, _ = simple_cull(costs, dominates, mtol=1.0)
        i, survival, distance = pareto_points.pop()
        p = ps[i]
    else:
        p = ps[0]
        attrs = attrss[0]
        survival = sum(a['survival'] for a in attrs)
        distance = sum(a[key] for a in attrs)
    return p, survival, distance


def _f(g, s, t, k0, k1, r, tol=0.1, key='distance'):
    if k0 not in r:
        p0, s0, d0 = comb_path(g, k0, s, t, key=key)
        r[k0] = (p0, s0, d0, np.exp(-s0))
    else:
        p0, s0, d0, _ = r[k0]
    if k1 not in r:
        p1, s1, d1 = comb_path(g, k1, s, t, key=key)
        r[k1] = (p1, s1, d1, np.exp(-s1))
    else:
        p1, s1, d1, _ = r[k1]
    if s1 and s0 / s1 < (1 + tol) and d0 and d1 / d0 < (1 + tol):
        return
    if np.isclose(s0, s1) or np.isclose(d1, d0):
        return
    a = (d1 - d0) / (s1 - s0)
    k = -a / (1 - a)
    if np.isclose(k, k0) or np.isclose(k, k1):
        return
    _f(g, s, t, k0, k, r, tol=tol)
    _f(g, s, t, k, k1, r, tol=tol)


def approx_pareto_convex_hul(g, s, t, tol=0.1, mtol=1.1):
    if not nx.has_path(g, s, t):
        return {}
    # print('has path')
    r = {}
    _f(g, s, t, 0, 1, r, tol=tol)
#     return r
    cs = [(k, d, s) for k, (_, s, d, _) in r.items()]
    # print('f', len(r))
    ncs, dom = simple_cull(cs, dominates, mtol=mtol)
    r1 = {k: r[k] for k, _, _ in ncs}
    rospy.loginfo('%d solutions reduced to %d', len(r), len(r1))
    return r1


def reach(graph, s):
    cols = int(np.sqrt(graph.number_of_nodes()))
    origin = nccr_origin(graph)
    stride = nccr_stride(graph, cols)
    s_node = nodeAtCoordinatates(s, origin, stride, cols)
    su = nx.shortest_path_length(graph, s_node, weight='survival')
    grid = np.array([su.get(node, np.inf)
                     for node in graph.nodes()]).reshape(cols, cols)
    return np.exp(-grid)


def plan(graph, s, t, tol=0.1, mtol=1.1):
    cols = int(np.sqrt(graph.number_of_nodes()))
    origin = nccr_origin(graph)
    stride = nccr_stride(graph, cols)
    s_node = nodeAtCoordinatates(s, origin, stride, cols)
    t_node = nodeAtCoordinatates(t, origin, stride, cols)
    rs = approx_pareto_convex_hul(graph, s_node, t_node, tol=tol, mtol=mtol)
    return rs


def reach_image(rs, elevation):
    rospy.loginfo('rs %s, elevation %s', rs.shape, elevation.shape)
    rs = scipy.misc.imresize(rs, elevation.shape)
    rs = rs
    img = np.stack([elevation, elevation, elevation, rs], axis=2)
    return img


def path(g, sol, tol=0.01):
    nodes = sol[0]
    ps = np.array([g.pos[node] for node in nodes])
    line = LineString(ps)
    line = line.simplify(tol)
    return np.array(line)


def pose(x, y, z=0):
    msg = PoseStamped()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    return msg


def color_from_traversability(f):
    return ColorRGBA(1 - f, f, 0, 1)


class Planner(object):
    """docstring for PathFollower."""

    def __init__(self):

        rospy.init_node("path_follower")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        path = rospy.get_param('~elevation_path')
        img = (plt.imread(path) * 255).astype(int)
        if len(img.shape) > 2:
            img = img[..., 0]
        self.elevation = img
        graph_path = rospy.get_param("~graph_path")
        max_traversability = rospy.get_param("~max_traversability", 0.99)
        G = nx.read_graphml(graph_path)
        self.graph = nccr_graph(G, max_traversability=max_traversability)
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.mtol = rospy.get_param("~mtol", 1.1)
        self.tol = rospy.get_param("~tol", 0.1)

        self.paths_pub = rospy.Publisher("paths", Paths, queue_size=1, latch=True)
        self.reach_pub = rospy.Publisher("reach", Image, queue_size=1, latch=True)
        rospy.Subscriber("target", PoseStamped, self.has_updated_target)
        rospy.Subscriber("pose", PoseStamped, self.has_updated_pose)
        rospy.spin()

    def has_updated_pose(self, msg):
        self.pose = pose_in_frame(self.tf_buffer, msg, self.frame_id)

    def has_updated_target(self, msg):
        target_pose = pose_in_frame(self.tf_buffer, msg, self.frame_id)
        s = array_from_msg(self.pose.pose.position)[:2]
        t = array_from_msg(target_pose.pose.position)[:2]
        rs = reach(self.graph, s)
        sols = plan(self.graph, s, t, mtol=self.mtol)

        paths_msg = Paths()

        # rospy.loginfo('sols %s', sols)

        for _, sol in sols.items():
            m = Path()
            paths_msg.paths.append(m)
            ps = path(self.graph, sol, tol=self.tol).tolist()
            m.path.header.frame_id = self.frame_id
            m.path.poses = [pose(*p) for p in ps]
            t = float(sol[3])
            le = float(sol[2])
            m.color = color_from_traversability(t)
            m.description = ["Traversability: {t:.0f}%".format(t=100 * t),
                             "Length: {le:.1f} m".format(le=le)]

        self.paths_pub.publish(paths_msg)

        image_msg = Image()
        image = reach_image(rs, self.elevation)
        # rospy.loginfo('shape %s', image.shape)
        image_msg.width, image_msg.height, n = image.shape
        image_msg.data = image.flatten().tolist()
        image_msg.encoding = 'rgba8'
        image_msg.step = n * image_msg.width

        self.reach_pub.publish(image_msg)


if __name__ == '__main__':
    Planner()
