from unittest.mock import patch, MagicMock

from server.external.weather_open_meteo import OpenMeteoWeather


def _stub_response(json_payload, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_payload
    r.raise_for_status = MagicMock()
    return r


def test_current_returns_spoken_summary():
    geocode = _stub_response({
        "results": [{"latitude": 38.8, "longitude": -77.1, "name": "Arlington"}]
    })
    forecast = _stub_response({
        "current": {"temperature_2m": 18.3, "weather_code": 2},
    })
    with patch("server.external.weather_open_meteo.requests.get",
               side_effect=[geocode, forecast]):
        w = OpenMeteoWeather()
        out = w.current("Arlington, VA")
    assert "Arlington" in out
    assert "F" in out
    assert any(word in out.lower() for word in ("cloud", "partly"))


def test_current_handles_unknown_location():
    geocode = _stub_response({"results": []})
    with patch("server.external.weather_open_meteo.requests.get",
               return_value=geocode):
        w = OpenMeteoWeather()
        out = w.current("Atlantis")
    assert "atlantis" in out.lower() or "couldn't find" in out.lower() \
        or "unknown" in out.lower()


def test_current_handles_api_failure_gracefully():
    geocode = _stub_response({"results": [{"latitude": 0, "longitude": 0,
                                           "name": "X"}]})
    failed = MagicMock()
    failed.raise_for_status.side_effect = Exception("502")
    with patch("server.external.weather_open_meteo.requests.get",
               side_effect=[geocode, failed]):
        w = OpenMeteoWeather()
        out = w.current("X")
    assert "weather" in out.lower()
