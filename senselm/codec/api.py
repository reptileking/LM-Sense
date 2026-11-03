"""One object holding the codec tables, the encoder and the decoder."""

from __future__ import annotations

from typing import Any

from senselm.codec.decoder import Decoder
from senselm.codec.encoder import Encoder, Trace
from senselm.codec.tables import load_tables


class Codec:
    def __init__(self, cfg: dict[str, Any], *, encoder: bool = True) -> None:
        conv = cfg.get("conventions", {})
        self.tables = load_tables(cfg, for_encoding=encoder)
        self.decoder = Decoder.from_tables(self.tables, auto_cap=bool(conv.get("auto_cap", False)))
        self.encoder: Encoder | None = None
        if encoder:
            import spacy

            nlp = spacy.load(cfg.get("spacy_model", "en_core_web_sm"), disable=["ner", "parser"])
            nlp.add_pipe("sentencizer")
            self.encoder = Encoder.from_tables(
                self.tables,
                nlp,
                auto_cap=bool(conv.get("auto_cap", False)),
                max_mwe_tokens=int(conv.get("max_mwe_tokens", 8)),
            )

    def encode(self, text: str, trace: Trace | None = None) -> list[int]:
        assert self.encoder is not None
        return self.encoder.encode(text, trace)

    def decode(self, ids: list[int]) -> str:
        return self.decoder.decode(ids)

    def names(self, ids: list[int]) -> list[str]:
        """Readable form: structural tokens by name, sense tokens as lemma#index/UPOS."""
        s, lex = self.tables["structural"], self.tables["lexicon"]
        out = []
        for t in ids:
            info = s.info.get(int(t))
            if info is not None:
                cat, name = info
                out.append(f"<{name}>" if cat == "special" else f"<{cat}:{name}>")
                continue
            hit = lex.lookup(int(t))
            out.append(f"{hit[0]}#{int(t) & 0x3FF}/{hit[1]}" if hit else f"<?{t}>")
        return out
