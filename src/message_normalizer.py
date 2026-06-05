from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


MEDIA_PLACEHOLDERS = {
    "messagePhoto": "[фото]",
    "messageVideo": "[видео]",
    "messageAnimation": "[анимация]",
    "messageDocument": "[файл]",
    "messageAudio": "[аудио]",
    "messageVoiceNote": "[голосовое сообщение]",
    "messageVideoNote": "[видеосообщение]",
    "messageContact": "[контакт]",
    "messageLocation": "[геолокация]",
    "messageVenue": "[место]",
    "messagePoll": "[опрос]",
}


def sender_identifier(sender: dict[str, Any] | None) -> str | None:
    if not sender:
        return None

    sender_type = sender.get("@type")

    if sender_type == "messageSenderUser":
        return f"user{sender.get('user_id')}"

    if sender_type == "messageSenderChat":
        return f"chat{sender.get('chat_id')}"

    return None


def normalize_reaction_type(
    reaction_type: dict[str, Any],
) -> dict[str, Any]:
    type_name = reaction_type.get("@type")

    if type_name == "reactionTypeEmoji":
        emoji = str(reaction_type.get("emoji", ""))

        return {
            "key": f"emoji:{emoji}",
            "type": "emoji",
            "emoji": emoji,
            "custom_emoji_id": None,
        }

    if type_name == "reactionTypeCustomEmoji":
        custom_emoji_id = str(
            reaction_type.get("custom_emoji_id", "")
        )

        return {
            "key": f"custom_emoji:{custom_emoji_id}",
            "type": "custom_emoji",
            "emoji": None,
            "custom_emoji_id": custom_emoji_id,
        }

    if type_name == "reactionTypePaid":
        return {
            "key": "paid",
            "type": "paid",
            "emoji": None,
            "custom_emoji_id": None,
        }

    serialized = json.dumps(
        reaction_type,
        ensure_ascii=False,
        sort_keys=True,
    )

    return {
        "key": f"unknown:{serialized}",
        "type": "unknown",
        "emoji": None,
        "custom_emoji_id": None,
    }


def extract_message_reactions(
    interaction_info: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not interaction_info:
        return []

    reactions_info = interaction_info.get("reactions")

    if not reactions_info:
        return []

    normalized_reactions: list[dict[str, Any]] = []

    for reaction in reactions_info.get("reactions", []):
        normalized_type = normalize_reaction_type(
            reaction.get("type", {})
        )

        normalized_reactions.append(
            {
                **normalized_type,
                "total_count": int(
                    reaction.get("total_count", 0)
                ),
                "is_chosen": bool(
                    reaction.get("is_chosen", False)
                ),
                "used_sender_id": sender_identifier(
                    reaction.get("used_sender_id")
                ),
                "recent_sender_ids": [
                    sender_identifier(sender)
                    for sender in reaction.get(
                        "recent_sender_ids",
                        [],
                    )
                    if sender_identifier(sender) is not None
                ],
            }
        )

    return sorted(
        normalized_reactions,
        key=lambda item: item["key"],
    )


def extract_formatted_text(
    formatted_text: dict[str, Any] | None,
) -> tuple[str, list[dict[str, Any]]]:
    if not formatted_text:
        return "", []

    text = str(formatted_text.get("text", "")).strip()
    entities = formatted_text.get("entities", [])

    return text, entities


def extract_message_content(
    content: dict[str, Any],
) -> tuple[
    str,
    list[dict[str, Any]],
    str,
    dict[str, Any],
] | None:
    content_type = content.get("@type")

    if content_type == "messageText":
        text, entities = extract_formatted_text(
            content.get("text")
        )

        if not text:
            return None

        return text, entities, content_type, {}

    if content_type == "messageAnimatedEmoji":
        emoji = str(content.get("emoji", "")).strip()

        if not emoji:
            return None

        return emoji, [], content_type, {}

    if content_type == "messageDice":
        emoji = str(content.get("emoji", "")).strip()

        if not emoji:
            return None

        return emoji, [], content_type, {
            "dice_value": int(content.get("value", 0)),
        }

    if content_type == "messageSticker":
        sticker = content.get("sticker", {})
        emoji = str(sticker.get("emoji", "")).strip()

        if emoji:
            text = f"[стикер: {emoji}]"
        else:
            text = "[стикер]"

        sticker_metadata = {
            "id": str(sticker.get("id", "")),
            "set_id": str(sticker.get("set_id", "")),
            "emoji": emoji or None,
            "width": int(sticker.get("width", 0)),
            "height": int(sticker.get("height", 0)),
            "format": (
                sticker.get("format", {}).get("@type")
            ),
            "full_type": (
                sticker.get("full_type", {}).get("@type")
            ),
            "is_premium": bool(
                content.get("is_premium", False)
            ),
        }

        return text, [], content_type, {
            "sticker": sticker_metadata,
        }

    if content_type in MEDIA_PLACEHOLDERS:
        placeholder = MEDIA_PLACEHOLDERS[content_type]
        caption, entities = extract_formatted_text(
            content.get("caption")
        )

        if caption:
            text = f"{placeholder}\n{caption}"
        else:
            text = placeholder

        return text, entities, content_type, {}

    return None


def normalize_text_message(
    message: dict[str, Any],
    tdlib_chat_id: int,
    own_user_id: int,
    own_name: str,
    other_user_id: int,
    other_name: str,
) -> dict[str, Any] | None:
    content = message.get("content", {})
    extracted = extract_message_content(content)

    if extracted is None:
        return None

    text, text_entities, content_type, extra_fields = extracted

    sender = message.get("sender_id", {})
    sender_type = sender.get("@type")

    sender_user_id = (
        sender.get("user_id")
        if sender_type == "messageSenderUser"
        else None
    )

    if sender_user_id == own_user_id:
        sender_name = own_name
    elif sender_user_id == other_user_id:
        sender_name = other_name
    else:
        sender_name = sender_identifier(sender) or "Unknown"

    date_unixtime = int(message["date"])

    date_iso = datetime.fromtimestamp(
        date_unixtime,
        tz=timezone.utc,
    ).isoformat().replace("+00:00", "Z")

    normalized = {
        "id": int(message["id"]),
        "type": "message",
        "content_type": content_type,
        "date": date_iso,
        "date_unixtime": date_unixtime,
        "from": sender_name,
        "from_id": sender_identifier(sender),
        "is_outgoing": bool(message.get("is_outgoing")),
        "text": text,
        "text_entities": text_entities,
        "reactions": extract_message_reactions(
            message.get("interaction_info")
        ),
        "tdlib_chat_id": tdlib_chat_id,
    }

    normalized.update(extra_fields)

    return normalized
