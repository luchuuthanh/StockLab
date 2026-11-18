import unittest

import pandas as pd

from growth_counter import calculate_growth_sequences


def frame(rows):
    return pd.DataFrame(rows, columns=["date", "symbol", "close", "EMA7", "SMA20"])


class GrowthCounterTests(unittest.TestCase):
    def test_day_one_continue_on_either_average_and_exclude_stop_day(self):
        data = frame([
            ("2026-11-02", "AAA", 8, 9, 10),   # Day 1: below both
            ("2026-11-03", "AAA", 9, 9, 10),   # equal EMA7: continue
            ("2026-11-04", "AAA", 10, 9, 10),  # equal SMA20: continue
            ("2026-11-05", "AAA", 8, 9, 10),   # stop; not counted
        ])
        result = calculate_growth_sequences(data)
        self.assertEqual(len(result["completed"]), 1)
        self.assertEqual(result["completed"][0]["Số ngày"], 3)
        self.assertEqual(str(result["completed"][0]["Ngày kết thúc"].date()), "2026-11-04")
        self.assertEqual([item["Số ngày"] for item in result["detail"]], [1, 2, 3])
        self.assertEqual(result["current"], [])

    def test_day_one_requires_close_below_both(self):
        data = frame([
            ("2026-11-02", "AAA", 9, 8, 10),  # above EMA7, below SMA20
            ("2026-11-03", "AAA", 10, 9, 11),
        ])
        result = calculate_growth_sequences(data)
        self.assertEqual(result["current"], [])
        self.assertEqual(result["completed"], [])
        self.assertEqual(result["detail"], [])

    def test_active_sequence_is_current_on_latest_market_date(self):
        data = frame([
            ("2026-11-02", "AAA", 8, 9, 10),
            ("2026-11-03", "AAA", 10, 9, 11),
            ("2026-11-04", "AAA", 12, 10, 11),
        ])
        result = calculate_growth_sequences(data)
        self.assertEqual(len(result["current"]), 1)
        self.assertEqual(result["current"][0]["Số ngày"], 3)

    def test_missing_intervening_market_session_interrupts_sequence(self):
        data = frame([
            ("2026-11-02", "AAA", 8, 9, 10),
            ("2026-11-03", "BBB", 10, 9, 11),  # market date; AAA missing
            ("2026-11-04", "AAA", 12, 10, 11),
            ("2026-11-04", "BBB", 12, 10, 11),
        ])
        result = calculate_growth_sequences(data)
        aaa = [item for item in result["interrupted"] if item["Mã"] == "AAA"]
        self.assertEqual(len(aaa), 1)
        self.assertEqual(aaa[0]["Trạng thái"], "Gián đoạn · thiếu phiên dữ liệu")
        self.assertEqual(result["current"], [])

    def test_ticker_missing_latest_market_date_is_not_marked_current(self):
        data = frame([
            ("2026-11-02", "AAA", 8, 9, 10),
            ("2026-11-03", "BBB", 10, 9, 11),
        ])
        result = calculate_growth_sequences(data)
        aaa = [item for item in result["interrupted"] if item["Mã"] == "AAA"]
        self.assertEqual(len(aaa), 1)
        self.assertEqual(aaa[0]["Trạng thái"], "Chưa xác nhận · mã thiếu phiên mới nhất")
        self.assertEqual(result["current"], [])

    def test_missing_required_column_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Thiếu cột lịch sử"):
            calculate_growth_sequences(pd.DataFrame({"date": [], "symbol": []}))


if __name__ == "__main__":
    unittest.main()
