from app.app import app


def test_predict_get():
    response = app.test_client().get("/predict")
    assert response.status_code == 200


def test_malformed_post_returns_400():
    response = app.test_client().post("/predict", data={"temp": "not-a-number"})
    assert response.status_code == 400
    assert b"must be numeric" in response.data


def test_valid_post_returns_prediction():
    response = app.test_client().post(
        "/predict",
        data={
            "temp": "29", "humidity": "74", "cloud_cover": "36",
            "rain_janfeb": "20.5", "rain_mar_may": "342",
            "rain_octdec": "501.5", "avg_june": "211.03", "sub": "430.6",
        },
    )
    assert response.status_code == 200
    assert b"chance of a flood" in response.data
