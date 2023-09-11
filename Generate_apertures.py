import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import product


def generate_coordinates(D, n=16, edge=False):
    """Generate coordinates  and size for optimal packing of n disks
    inscribed within larger disk of diameter D"""

    points = np.zeros((n, 2))

    def ring(r, N, range=2 * np.pi, offset=0):
        theta = (range * np.arange(N) / N) + offset

        return r * np.stack([np.cos(theta), np.sin(theta)], axis=1)

    def interior_points(D, n, edge=False):
        """Ring of n disks around outer radius D"""
        ct = np.cos((n - 2) / n / 2 * np.pi)
        d = D / ((not edge) + 1 / ct)
        points = ring(d / 2 / ct, n)
        return points, d

    if n == 1:
        d = D
    if n > 1 and n < 7:
        points, d = interior_points(D, n, edge)
    if n > 6 and n < 10:
        points, d = interior_points(D, n - 1, edge)
        points = np.concatenate([points, [[0, 0]]], axis=0)
    if n == 10:
        f = 3.813
        d = D / (f - edge)
        theta_step = 2 * np.arcsin(d / (D - d * (not edge)))
        theta0 = (2 * np.pi - theta_step * (n - 3)) / 2
        points[:8] = ring(
            D / 2 - d * (not edge) / 2, n - 2, theta_step * (n - 2), theta0
        )
        points[8, 0] = points[(n - 2) // 2, 0] + 2 * d * np.cos(theta_step / 2)
        points[9, 0] = points[8, 0] - d
    if n == 11:
        points, d = interior_points(D, n - 2, edge)
        points = np.concatenate(
            [
                points,
                [[-d / 2, 0], [d / 2 * np.cos(np.pi / 9), d / 2 * np.sin(np.pi / 9)]],
            ]
        )
    if n == 12:
        f = 4.029
        d = D / (f - edge)
        theta_step = 2 * np.arcsin(d / (D - d * (not edge)))
        boundary = (2 * np.pi - 6 * theta_step) / 3
        for i in range(3):
            theta0 = boundary * i + theta_step * 2 * i
            points[i * 3 : (i + 1) * 3] = ring(
                D / 2 - d * (not edge) / 2, 3, theta_step * 3, theta0
            )
        points[-3:] = ring(d / 2 / np.cos(np.pi / 6), 3, offset=-boundary / 2)
    if n == 13:
        points, d = interior_points(D, n - 3, edge)
        points = np.concatenate([points, ring(d / 2 / np.cos(np.pi / 6), 3)])
    if n == 14:
        f = 4.328
        d = D / (f - edge)
        theta_step = 2 * np.arcsin(d / (D - d * (not edge)))
        theta0 = (2 * np.pi - theta_step * (n - 5)) / 2
        points[: n - 4] = ring(
            D / 2 - d * (not edge) / 2, n - 4, theta_step * (n - 4), theta0
        )
        # x position of 11th circle
        points[-4, 0] = points[0, 0] - np.sqrt(d**2 - points[0, 1] ** 2)
        # x position of 13th circle
        points[-2, 0] = points[5, 0] + np.sqrt(d**2 - points[5, 1] ** 2)
        # x position of 12th and 14th circles
        points[-3::2, 0] = (points[-4, 0] + points[-2, 0]) / 2
        # y position of 12th and 14th circles
        points[-3::2, 1] = [
            ((-1) ** n) * np.sqrt(d**2 - (points[-4, 0] - points[-2, 0]) ** 2 / 4)
            for n in range(2)
        ]
    if n == 15:
        f = 1 + np.sqrt(6 + 2 / np.sqrt(5) + 4 * np.sqrt(1 + 2 / np.sqrt(5)))  # +edge
        d = D / (f - edge)
        theta_step = 2 * np.arcsin(d / (D - d * (not edge)))
        boundary = (2 * np.pi - 5 * theta_step) / 5
        for i in range(5):
            theta0 = boundary * (i + 0.5) + theta_step * i
            points[i * 2 : (i + 1) * 2] = ring(
                D / 2 - d * (not edge) / 2, 2, theta_step * 2, theta0
            )
        points[-5:] = ring(d / 2 / np.sin(np.pi / 5), 5, offset=0)
 
    if n == 16:
        f = 4.615
        # Diameter of inscribed disks
        d = D / (f - edge)

        # Calculate positions of exterior points
        theta_step = 2 * np.arcsin(d / (D - d * (not edge)))
        theta0 = (2 * np.pi - theta_step * 10) / 2
        theta = np.arange(11) * theta_step + theta0
        # points[5:] = D / 2 * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        points[5:] = ring(
            D / 2 - d * (not edge) / 2, n - 5, theta_step * (n - 5), theta0
        )
        # calculate positions of interior points
        # distance of points from pentagon centre
        points[:5] = ring(d / 2 / np.cos(3 * np.pi / 10), 5)  # ,2*np.pi/5)
        # Pentagon shifted to right by small amount
        dx = (points[-1, 0] - points[0, 0]) - np.sqrt(d**2 - points[-1, 1] ** 2)
        points[:5, 0] += dx
    return points, d


def rot_matrix(angle, degrees=True):
    """Make 2 x 2 rotation matrix for rotation by angle in either degrees
    or radians."""
    if degrees:
        a = np.deg2rad(angle)
    else:
        a = angle
    ct, st = [np.cos(a), np.sin(a)]
    return np.asarray([[ct, st], [-st, ct]])


def rotated_rectangle(dimensions, rotation, closed=True):
    """Generate coordinates for a rotated rectangle of given dimensions."""

    if len(dimensions) > 1:
        d = dimensions
    else:
        d = dimensions * np.ones(2)
    
    points = np.asarray(
        [
            [
                d[0] / 2 * ((-1) ** ((i + 1) // 2)),
                d[1] / 2 * ((-1) ** (i // 2)),
            ]
            for i in range(4 + closed)
        ]
    )
    
    return (rot_matrix(rotation) @ points.T).T


def polygons_to_dxf(polygons, fnam):
    """Convert a set of polygons to a CAD dxf object."""
    import ezdxf

    doc = ezdxf.new("R2000")
    msp = doc.modelspace()
    for i, polygon in enumerate(polygons):
        msp.add_lwpolyline(list(map(tuple, polygon)))
    doc.saveas(fnam)


def generate_aperture_plate(sizes,rotations, D, edge=True, rotation=None, scaling=1):
    """Given a list containing the sizes and rotations of apertures
    produce list of polygons for aperturesplate

    Parameters
    ----------
    size_and_rotations : (n,2) or (n,3) array_like
        The dimensions and then rotations of the apertures, pass (n,2) for square apertures
        in format [rotation,width] and (n,3) for rectangular apertures in format [rotation, width, height]
    D : float scalar
        Diameter of aperture plate
    Returns
    -------
    polygons : list of (4,) nd_array
        Coordinates of aperture vertices
    d : scalar
        minimum seperation of aperture centres
    """

    sze = np.asarray(sizes)
    rtn =np.asarray(rotations) 

    n = sze.shape[0]

    centres, d = generate_coordinates(D, n, edge)

    polygons = []

    for centre, app, rot in zip(centres, sze,rtn):
        # print(centre,rotated_rectangle(app[1:] * scaling, app[0]).shape)
        polygons.append(rotated_rectangle(app * scaling, rot) + centre)

    if rotation is not None:
        R = rot_matrix(rotation)
        for i in range(len(polygons)):
            polygons[i] = (R @ polygons[i].T).T
        for i in range(len(centres)):
            centres[i] = (R @ centres[i].T).T

    return polygons, d, centres


def plot_apertures(polygons, centres=None, d=None, D=None, ax=None, figsize=(4, 4)):
    from matplotlib.patches import Circle, Rectangle

    if ax is None:
        returnfig = True
        fig, ax = plt.subplots(figsize=figsize)
    else:
        returnfig = False
    ax.add_patch(Circle([0, 0], radius=3040 / 2, ec="k", fc="#f2ce40ff"))
    for i, poly in enumerate(polygons):
        ax.plot(poly[:, 0], poly[:, 1], "k-", linewidth=0.25)
        if centres is not None and d is not None:
            ax.annotate(
                str(i + 1), centres[i] + d * np.sqrt(2) / 8 * np.asarray([1, 1])
            )

    if d is not None and centres is not None:
        xy = (centres[-1] + centres[-2]) / 2
        diff = centres[-1] - centres[-2]
        theta = np.rad2deg(np.arctan(diff[-1] / diff[-2]))
    if D is not None:
        ax.add_patch(Circle([0, 0], radius=D / 2, ec="k", fc="none", linestyle=":"))

    ax.set_ylabel("y (\u03BCm)")
    ax.set_xlabel("x (\u03BCm)")
    # fig.tight_layout()
    if returnfig:
        return fig


if __name__ == "__main__":
    if not os.path.exists("Apertures"):
        os.mkdir("Apertures")

    # Custom aperture plates
    # Put rotations in degrees here
    rotations = np.asarray([0,20,40,60,80]*3)
    # Put sizes of aperture plates (in microns) here
    sizes = np.asarray([20,40,60]*3)
    # For rectangles you need to put both dimensions as a list eg:
    sizes = np.asarray([[20,30]]*5+[[40,50]]*5+[[60,70]]*5)
    print(sizes)
    #File name for output
    fnam = "Apertures/Demonstration"
    # The diameter (in micron) within which to fit the apertures (either 1.2 or 2 mm)
    D = 1.2e3

    # sizes_and_rotations = np.concatenate((rotations, sizes),axis=1)
    polygons, d, centres = generate_aperture_plate(
        sizes,rotations, D - 1e2, True, scaling=1
    )
    polygons_to_dxf(polygons, "Demonstration.dxf")

    polygons, d, centres = generate_aperture_plate(
        sizes,rotations, D - 1e2, True, scaling=1
    )
    fig = plot_apertures(polygons, centres, d, D)
    
    fig.savefig(fnam + ".pdf")
    fig.savefig(fnam + ".png")
    plt.show(block=True)
    import sys;sys.exit()
    import os

    
    for iD, D in enumerate([1.2e3, 1.8e3]):
        edge = True
        # MCN apertures
        sizes = [24, 19.5, 16, 13.5]
        scaling = 2
        rotations = 15 * np.arange(4)
        
        # Norcada apertures
        # sizes = [32.5,22.5]
        # rotations = 15 * np.arange(6)
        sizes_and_rotations = np.asarray(list(product(rotations, sizes)))
        polygons, d, centres = generate_aperture_plate(
            sizes_and_rotations[:,:2],sizes_and_rotations[:,2], D - 1e2, edge, scaling=scaling
        )
        fig = plot_apertures(polygons, centres, d, D)
        fnam = "Apertures/Arctica_{0}".format(2 * iD)
        fig.savefig(fnam + ".pdf")
        fig.savefig(fnam + ".png")
        polygons, d, centres = generate_aperture_plate(
            sizes_and_rotations, D - 1e2, edge
        )
        polygons_to_dxf(polygons, fnam + ".dxf")

        # sizes = [29.5]
        # rotations = 15 * np.arange(4)
        # sizes_and_rotations = np.asarray(list(product(rotations,sizes)))
        # polygons,d,centres= generate_aperture_plate(sizes_and_rotations[:,:2],sizes_and_rotations[:,2],D-1e2,edge)
        # fig = plot_apertures(polygons,centres,d,D)
        # fnam = "Apertures/Arctica_{0}".format(2*iD+1)
        # fig.savefig(fnam+'.pdf')
        # polygons_to_dxf(polygons, fnam+".dxf")

        # MCN apertures
        # K3 rectangles
        n = 12
        sizes = np.repeat(np.asarray([70, 50]).reshape((1, 2)), n, axis=0)
        rotations = 15 * np.arange(n).reshape((n, 1))
        sizes_and_rotations = np.concatenate((rotations, sizes), axis=1)
        # Falcon squares
        sizes = [[30, 30]]
        n = 4
        sizes = np.repeat(np.asarray([50, 50]).reshape((1, 2)), n, axis=0)
        rotations = 15 * np.arange(n).reshape((n, 1))

        # Norcada apertures
        # K3 rectangles
        # sizes = np.repeat(np.asarray([42,30]).reshape((1,2)),9,axis=0)
        # rotations = 20 * np.arange(9).reshape((9,1))
        # sizes_and_rotations = np.concatenate((rotations,sizes),axis=1)
        # #Falcon squares
        # sizes = [[30,30]]
        # sizes = np.repeat(np.asarray([30,30]).reshape((1,2)),6,axis=0)
        # rotations = 15 * np.arange(6).reshape((6,1))

        sizes_and_rotations = np.concatenate(
            (sizes_and_rotations[:,:2],sizes_and_rotations[:,2], np.concatenate((rotations, sizes), axis=1)), axis=0
        )
        polygons, d, centres = generate_aperture_plate(
            sizes_and_rotations[:,:2],sizes_and_rotations[:,2], D - 1e2, edge, rotation=200, scaling=1
        )
        fig = plot_apertures(polygons, centres, d, D)
        fnam = "Apertures/Krios_{0}".format(iD)
        fig.savefig(fnam + ".pdf")
        fig.savefig(fnam + ".png")
        polygons_to_dxf(polygons, fnam + ".dxf")

        # sizes_and_rotations = np.concatenate((rotations,sizes),axis=1)
        # polygons,d,centres= generate_aperture_plate(sizes_and_rotations,D-1e2,edge)
        # fig = plot_apertures(polygons,centres,d,D)
        # fnam = "Apertures/Krios_{0}".format(2*iD+2)
        # fig.savefig(fnam+'.pdf')
        # polygons_to_dxf(polygons, fnam+".dxf")
    plt.show(block=True)
