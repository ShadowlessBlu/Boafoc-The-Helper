import os
import tempfile
import unittest

import _path  # noqa: F401
import json

from assistant.memory import Memory, cosine, decode_vec, encode_vec


def toy_embedder(texts):
    """Bag-of-letters vectors: enough to test the cosine path without a model."""
    out = []
    for t in texts:
        v = [0.0] * 26
        for ch in t.lower():
            if "a" <= ch <= "z":
                v[ord(ch) - 97] += 1
        out.append(v)
    return out


class MemoryTests(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)

    def test_messages_order(self):
        m = Memory()
        for i in range(5):
            m.add_message("user", f"msg {i}")
        self.assertEqual([x["content"] for x in m.recent_messages(3)], ["msg 2", "msg 3", "msg 4"])

    def test_keyword_search(self):
        m = Memory()
        m.remember("The user's favourite colour is green.")
        m.remember("The user's sister was born on 4 March.")
        hits = m.search("what is my favourite colour?")
        self.assertTrue(hits)
        self.assertIn("green", hits[0][2])

    def test_embedding_search(self):
        m = Memory(embedder=toy_embedder)
        m.remember("zzz quiet")
        m.remember("favourite colour green")
        hits = m.search("favourite colour", top_k=1, min_score=0.0)
        self.assertIn("colour", hits[0][2])

    def test_vectors_stored_as_float32_blobs(self):
        m = Memory(embedder=toy_embedder)
        m.remember("favourite colour green")
        (emb,) = m.db.execute("SELECT embedding FROM memories").fetchone()
        self.assertIsInstance(emb, bytes)
        self.assertEqual(len(emb), 26 * 4)
        self.assertAlmostEqual(decode_vec(emb)[0], toy_embedder(["favourite colour green"])[0][0], places=6)
        self.assertEqual(decode_vec(encode_vec([0.5, -1.0])), [0.5, -1.0])

    def test_old_json_rows_still_searchable(self):
        """Databases written before the blob change keep working, even mixed with new rows."""
        m = Memory(embedder=toy_embedder)
        old = json.dumps(toy_embedder(["favourite colour green"])[0])
        m.db.execute("INSERT INTO memories (ts, text, embedding) VALUES (0, 'favourite colour green', ?)", (old,))
        m.remember("zzz quiet")
        hits = m.search("favourite colour", top_k=1, min_score=0.0)
        self.assertIn("colour", hits[0][2])

    def test_search_sees_new_and_forgotten_memories(self):
        """The cached numpy matrix must be rebuilt when memories are added or removed."""
        m = Memory(embedder=toy_embedder)
        m.remember("zzz quiet")
        m.search("anything", min_score=0.0)
        mid = m.remember("favourite colour green")
        self.assertIn("colour", m.search("favourite colour", top_k=1, min_score=0.0)[0][2])
        m.forget(mid)
        self.assertNotIn("colour", m.search("favourite colour", top_k=1, min_score=0.0)[0][2])

    def test_forget(self):
        m = Memory()
        mid = m.remember("temporary")
        self.assertTrue(m.forget(mid))
        self.assertFalse(m.forget(mid))
        self.assertEqual(m.list_memories(), [])

    def test_audit_and_file_db(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "a.db")
            m = Memory(path)
            m.log_audit("set_light", {"room": "kitchen"}, "allow", "low action", "ok")
            m.db.close()
            m2 = Memory(path)
            self.assertEqual(m2.audit_log()[0][1], "set_light")
            m2.db.close()


if __name__ == "__main__":
    unittest.main()
