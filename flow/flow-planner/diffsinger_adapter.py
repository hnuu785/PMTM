"""DiffSinger 연동을 위한 공통 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class KoreanG2PAdapter(ABC):
    """음절을 모델별 음소로 변환한다."""

    @abstractmethod
    def to_phonemes(
        self,
        syllable: dict[str, Any],
        context: Sequence[dict[str, Any]],
    ) -> list[str]:
        """음절 하나의 음소를 반환한다."""


class DurationAllocator:
    """음소 길이의 합을 음절 길이와 맞춘다."""

    def allocate(
        self,
        duration_slots: int,
        phonemes: Sequence[str],
        has_onset: bool,
        has_coda: bool,
    ) -> list[int]:
        if duration_slots < 0:
            raise ValueError("duration_slots must be non-negative")
        if not phonemes:
            return []

        if has_onset and has_coda:
            weights = [0.2, 0.6, 0.2]
        elif has_onset:
            weights = [0.25, 0.75]
        elif has_coda:
            weights = [0.75, 0.25]
        else:
            weights = [1.0]

        weights = (weights + [0.0] * len(phonemes))[: len(phonemes)]
        if not any(weights):
            weights[0] = 1.0
        weight_sum = sum(weights)
        raw = [duration_slots * weight / weight_sum for weight in weights]
        allocated = [int(value) for value in raw]

        for index in sorted(
            range(len(allocated)), key=lambda item: raw[item] - allocated[item], reverse=True
        )[: duration_slots - sum(allocated)]:
            allocated[index] += 1
        return allocated


class DiffSingerInputBuilder:
    """플로우 플랜을 공통 음표·쉼표 이벤트로 변환한다."""

    def __init__(self, g2p: KoreanG2PAdapter | None = None) -> None:
        self.g2p = g2p

    def build_common_events(self, flow_plan: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in flow_plan.get("lines", []):
            context = [
                syllable
                for segment in line.get("segments", [])
                for syllable in segment.get("syllables", [])
            ]
            for segment in line.get("segments", []):
                for syllable in segment.get("syllables", []):
                    phonemes = self.g2p.to_phonemes(syllable, context) if self.g2p else []
                    events.append({
                        "type": "note",
                        "syllable": syllable["text"],
                        "phonemes": phonemes,
                        "phoneme_durations": [],
                        "midi_note": syllable["midi_note"],
                        "is_slur": syllable["is_slur"],
                        "start_sec": syllable["start_sec"],
                        "end_sec": syllable["end_sec"],
                    })
            events.extend({
                "type": "rest",
                "start_sec": rest["start_sec"],
                "end_sec": rest["end_sec"],
            } for rest in line.get("rests", []))
        return sorted(events, key=lambda event: event["start_sec"])
