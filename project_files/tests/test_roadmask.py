import numpy as np

from trafficflow.roadmask import inside_mask, load_road_polygons, points_inside

SQUARE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
OTHER = SQUARE + np.array([300, 0], dtype=np.float32)


def test_anchor_is_bottom_centre_not_box_centre():
    # box centre (50, 65) is inside, bottom-centre (50, 130) is outside
    assert not inside_mask(np.array([[30, 0, 70, 130]], dtype=np.float32), [SQUARE])[0]
    assert inside_mask(np.array([[30, 40, 70, 90]], dtype=np.float32), [SQUARE])[0]


def test_vectorised_result_matches_each_box():
    boxes = np.array([[10, 10, 30, 30], [200, 200, 220, 220], [60, 60, 80, 99]], dtype=np.float32)
    assert inside_mask(boxes, [SQUARE]).tolist() == [True, False, True]


def test_point_is_on_road_if_inside_any_polygon():
    pts = np.array([[50, 50], [350, 50], [200, 50]], dtype=np.float32)
    assert points_inside(pts, [SQUARE, OTHER]).tolist() == [True, True, False]


def test_config_polygons_load_as_list_of_arrays():
    polys = load_road_polygons()
    assert isinstance(polys, list) and all(p.ndim == 2 and p.shape[1] == 2 for p in polys)


def test_write_road_mask_roundtrip(tmp_path, monkeypatch):
    import trafficflow.roadmask as rm
    zones = tmp_path / "zones.yaml"
    zones.write_text("approaches: {}\n\n# Road mask: old\nroad_mask: []\n")
    rm.write_road_mask(zones, [[(0, 0), (10, 0), (10, 10)], [(100.4, 100.6), (120, 100), (120, 120), (100, 120)]])
    monkeypatch.setattr(rm, "load_config", lambda name: __import__("yaml").safe_load(zones.read_text()))
    polys = rm.load_road_polygons()
    assert [p.shape for p in polys] == [(3, 2), (4, 2)]
    assert polys[1][0].tolist() == [100.0, 101.0]
    assert "approaches" in zones.read_text()
