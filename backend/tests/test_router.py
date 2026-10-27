import unittest
from backend.router import ahp_weights, route_query
from backend.evaluation import routing_metrics

class RouterTests(unittest.TestCase):
    def test_ahp_weights_sum_to_one(self):
        weights, cr = ahp_weights()
        self.assertAlmostEqual(sum(weights), 1.0, places=6)
        self.assertLess(cr, 0.1)

    def test_cross_document_query_prefers_parallel(self):
        result = route_query("Compare information across documents", 0.2, 0.8, {"graph_entity_count": 2, "qdrant_point_count": 20})
        self.assertEqual(result["selected_strategy"], "way2")

    def test_relationship_query_can_prefer_graph_first(self):
        result = route_query("What is the relationship between Alice and Acme?", 0.95, 0.2, {"graph_entity_count": 30, "qdrant_point_count": 30}, True)
        self.assertEqual(result["selected_strategy"], "way1")

    def test_confusion_metrics(self):
        result = routing_metrics(["way1", "way2"], ["way1", "way1"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["confusion_matrix"]["way2"]["way1"], 1)

if __name__ == "__main__":
    unittest.main()
