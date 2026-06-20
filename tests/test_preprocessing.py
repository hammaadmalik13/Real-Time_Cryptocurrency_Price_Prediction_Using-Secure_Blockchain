from __future__ import annotations

import numpy as np

from crypto_lstm.preprocessing import create_sequences, split_train_test, target_index_for


def test_create_sequences_uses_target_index() -> None:
    data = np.array(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [3.0, 13.0],
        ]
    )

    X, y = create_sequences(data, window_size=2, target_index=1)

    assert X.shape == (2, 2, 2)
    assert y.tolist() == [12.0, 13.0]


def test_split_train_test_keeps_order() -> None:
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)

    X_train, X_test, y_train, y_test = split_train_test(X, y, train_ratio=0.6)

    assert X_train.tolist() == [[0, 1], [2, 3], [4, 5]]
    assert X_test.tolist() == [[6, 7], [8, 9]]
    assert y_train.tolist() == [0, 1, 2]
    assert y_test.tolist() == [3, 4]


def test_target_index_for_close() -> None:
    assert target_index_for(["open", "close", "volume"]) == 1

