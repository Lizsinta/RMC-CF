import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
from pyqtgraph.Qt import QtCore
import numpy as np
from math import sqrt, acos, pi
import matplotlib


class myPlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super(myPlotWidget, self).__init__(parent)


def psi(i, j, k):
    x = i - i.shape[0]/2
    y = j - i.shape[1]/2
    z = k
    r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    return r

def color_convert(c, a, octal=True) -> list:
    if type(c) == str:
        c = list(matplotlib.colors.to_rgb(c))
    if octal:
        a *= 255
        if max(c) <= 1.0:
            color = [c[0] * 255, c[1] * 255, c[2] * 255, a]
        else:
            color = [c[0], c[1], c[2], a]
    else:
        if max(c) <= 1.0:
            color = [c[0], c[1], c[2], a]
        else:
            color = [c[0] / 255, c[1] / 255, c[2] / 255, a]
    return color


def scatter(x=0, y=0, z='', c=(255., 255., 255.), edge_c=(255., 255., 255.), alpha=1.0, scale=1.0):
    if not type(z) == str:
        md = gl.MeshData.sphere(rows=10, cols=20)
        absorber = gl.GLMeshItem(meshdata=md, color=color_convert(c, alpha, octal=False),
                                 shader='shaded', glOptions='translucent')
        absorber.translate(x, y, z)
        absorber.scale(scale, scale, scale)
    else:
        absorber = pg.ScatterPlotItem(x=np.asarray(x), y=np.asarray(y), size=scale*10,
                                      pen={'color': color_convert(edge_c, alpha, octal=True)})
        absorber.setBrush(color=color_convert(c, alpha, octal=True))
    return absorber


def cylinder(x, y, z, c, radius=np.array([1, 1]), alpha=1., width=0.2):
    length = sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2 + (z[1] - z[0]) ** 2)
    md = gl.MeshData.cylinder(10, 20, radius=radius)
    bond = gl.GLMeshItem(meshdata=md, color=color_convert(c, alpha, octal=False),
                         shader='shaded', glOptions='translucent')
    bond.scale(width, width, length)
    angle = acos((z[1] - z[0]) / length) / pi * 180
    bond.rotate(angle, -(y[1] - y[0]), (x[1] - x[0]), 0)
    bond.translate(x[0], y[0], z[0])
    return bond


def line(x=np.array([]), y=np.array([]), z=np.array([]), c=(0, 0, 0), alpha=1.0, width=1.0):
    if len(z) > 0:
        pts = np.vstack([x, y, z]).transpose()
        return gl.GLLinePlotItem(pos=pts, color=color_convert(c, alpha, octal=False),
                                 width=width, antialias=True, glOptions='opaque')
    else:
        return pg.PlotDataItem(x, y, pen={'color': color_convert(c, alpha, octal=True), 'width': width})


def bar(x=np.array([]), y=np.array([]), width=0.3, c=(0, 0, 0), alpha=1.0):
    return pg.BarGraphItem(x=x, height=y, width=width, brush=color_convert(c, alpha, octal=True))


def plane(x=np.array([]), y=np.array([]), z=np.array([]), c=(0, 0, 0), alpha=1.0):
    return gl.GLSurfacePlotItem(x=x, y=y, z=z, shader='shaded', glOptions='translucent',
                                color=color_convert(c, alpha, octal=False))


def hemisphere(radius, c=(0, 0, 0), alpha=1.0):
    color = color_convert(c, alpha, octal=True)
    data = np.abs(np.fromfunction(psi, (20, 20, 20)))
    verts, faces = pg.isosurface(data, data.max() / 4.)
    md = gl.MeshData(vertexes=verts, faces=faces)
    colors = np.zeros((md.faceCount(), 4), dtype=float)
    colors[:, 0] = color[0]
    colors[:, 1] = color[1]
    colors[:, 2] = color[2]
    colors[:, 3] = alpha
    md.setFaceColors(colors)
    m2 = gl.GLMeshItem(meshdata=md, smooth=True, shader='shaded', glOptions='translucent')
    factor = 11.8
    diameter = 2 * radius
    m2.scale(diameter / factor, diameter / factor, diameter / factor)
    m2.translate(-diameter * 10 / factor, -diameter * 10 / factor, 0)
    return m2


def init_3dplot(target, grid=True, background=[255., 255., 255.], alpha=1.0, view=40, angle=[], title='', size=[1, 1, 1]):
    c = (np.zeros(3) + 255) - background
    offset = [-size[0] / 2, -size[1] / 2, size[2] / 2]
    if grid:
        gx = gl.GLGridItem()
        gx.setColor((*c, 100))
        gx.rotate(90, 1, 0, 0)
        gx.translate(offset[1]+size[1], size[0], offset[2])
        gx.setSize(size[1], size[2])
        target.addItem(gx)

        gy = gl.GLGridItem()
        gy.setColor((*c, 100))
        gy.rotate(90, 0, 1, 0)
        gy.translate(size[1], offset[0]+size[0], offset[2])
        gy.setSize(size[2], size[0])
        target.addItem(gy)

        gz = gl.GLGridItem()
        gz.setColor((*c, 100))
        gz.rotate(90, 0, 0, 1)
        gz.translate(offset[1]+size[1], offset[0]+size[0], 0)
        gz.setSize(size[0], size[1])
        target.addItem(gz)
    else:
        framework = np.array([
        line(np.array([0, size[0]]), np.array([0, 0]), np.array([0, 0]), c='r', alpha=1.0, width=1),
        line(np.array([0, size[0]]), np.array([0, 0]), np.array([size[2], size[2]]), c=c, alpha=1.0, width=1),
        line(np.array([0, size[0]]), np.array([size[1], size[1]]), np.array([0, 0]), c=c, alpha=1.0, width=1),
        line(np.array([0, size[0]]), np.array([size[1], size[1]]), np.array([size[2], size[2]]), c=c, alpha=1.0, width=1),
        line(np.array([0, 0]), np.array([0, size[1]]), np.array([0, 0]), c='g', alpha=1.0, width=1),
        line(np.array([0, 0]), np.array([0, size[1]]), np.array([size[2], size[2]]), c=c, alpha=1.0, width=1),
        line(np.array([size[0], size[0]]), np.array([0, size[1]]), np.array([0, 0]), c=c, alpha=1.0, width=1),
        line(np.array([size[0], size[0]]), np.array([0, size[1]]), np.array([size[2], size[2]]), c=c, alpha=1.0, width=1),
        line(np.array([0, 0]), np.array([size[1], size[1]]), np.array([0, size[2]]), c=c, alpha=1.0, width=1),
        line(np.array([0, 0]), np.array([0, 0]), np.array([0, size[2]]), c='b', alpha=1.0, width=1),
        line(np.array([size[0], size[0]]), np.array([size[1], size[1]]), np.array([0, size[2]]), c=c, alpha=1.0, width=1),
        line(np.array([size[0], size[0]]), np.array([0, 0]), np.array([0, size[2]]), c=c, alpha=1.0, width=1)])
        for i in framework:
            i.translate(offset[0], offset[1], offset[2])
            target.addItem(i)

    target.opts['distance'] = view
    if not len(angle) == 0:
        target.opts['azimuth'] = angle[0]
        target.opts['elevation'] = angle[1]
    else:
        target.opts['azimuth'] = 225
        target.opts['elevation'] = 30
    target.setBackgroundColor(color_convert(background, alpha))
    target.setWindowTitle(title)

class barPlotWidget(pg.PlotWidget):
    def __init__(self, parent=None, background=None):
        super().__init__(parent, background)
        self.autoBtn_flag = True
        self.getViewBox().sigRangeChangedManually.connect(self.user_behavior)
        #self.sigWheelEvent.connect(self.user_behavior)

    def auto_range(self):
        self.autoBtn_flag = True

    def user_behavior(self):
        self.autoBtn_flag = False


if __name__ == '__main__':
    from sys import flags
    from sys import argv, exit
    # framework: windows->large widget->layout->all kinds of graph widgets-> item
    app = QApplication([])
    mw = QMainWindow()
    mw.resize(1280, 720)
    mw.setWindowTitle('pyqtgraph example')
    mw.show()
    cw = QWidget()
    mw.setCentralWidget(cw)
    l = QHBoxLayout()
    cw.setLayout(l)

    # 3d graph
    g3d = gl.GLViewWidget()
    init_3dplot(g3d, background=[255., 255., 255.], alpha=1.0, view=40, title='example')
    l.addWidget(g3d)
    sc = scatter(0, 0, 0, c='red', alpha=1.0, scale=0.5)
    g3d.addItem(sc)
    cy = cylinder([0, 0], [0, 1], [0, 0], c='brown', alpha=1.0, width=0.2)
    g3d.addItem(cy)
    line3d = line(np.array([0, 2]), np.array([0, 2]), np.array([0, 2]), c='black', alpha=1.0, width=3)
    g3d.addItem(line3d)
    surface = plane(np.arange(10)-5, np.arange(10)-5, np.zeros((10, 10)), c=(127, 127, 255, 0.1))
    #g3d.addItem(surface)
    line3d.setData(pos=np.array([[2, 2, 2], [4, 4, 4]]))

    # 2d graph
    g2d = pg.PlotWidget(background=[255, 255, 255, 255])
    g2d.getAxis('left').hide()
    g2d.getAxis('bottom').hide()
    #g2d.setLabel('left', 'y', units='Y')
    #g2d.setLabel('bottom', 'x', units='X')
    #g2d.setXRange(0, 10)
    #g2d.setYRange(0, 11)
    l.addWidget(g2d)
    ba = bar(np.arange(10), np.arange(1, 11), c='blue', alpha=0.6, width=0.5)
    #g2d.addItem(ba)
    line2d = line(np.arange(10), np.arange(1, 11), c='cyan', alpha=1.0, width=2)
    g2d.addItem(line2d)
    line2d.setData(np.arange(10), 10 - np.arange(1, 11))


    g3d.setSizePolicy(g2d.sizePolicy())  # Priority of 2d graph is higher, it would squeeze out the 3d graph

    '''sc.setTransform()
    sc.translate(1, 1, 1)
    print(sc.viewTransform())
    sc.translate(-1, -1, -1, True)
    print(sc.viewTransform())'''

    if (flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QApplication.instance().exec()
