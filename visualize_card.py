import argparse
from typing import Optional

from cards import EnhancementEngine, CardState, load_cards


def pick_card(cards: list[CardState], card_id: Optional[str], card_type: str) -> CardState:
    if card_id:
        for c in cards:
            if c.id == card_id:
                return c
        raise ValueError(f"카드를 찾을 수 없습니다: id={card_id}")
    for c in cards:
        if c.type == card_type:
            return c
    raise ValueError(f"{card_type} 타입 카드를 찾을 수 없습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description="enhancements visual 설정으로 카드 이미지를 렌더링해 화면에 표시합니다.")
    parser.add_argument("--enhancements", default="enhancements.json", help="강화 설정 JSON 경로")
    parser.add_argument("--cards", default="cards.json", help="카드 JSON 경로")
    parser.add_argument("--card-id", default=None, help="렌더링할 카드 id")
    parser.add_argument("--card-type", default="attack", choices=["attack", "defense"], help="card-id 미지정 시 선택할 타입")
    parser.add_argument("--image-path", default=None, help="카드 아트 이미지 경로 override")
    parser.add_argument("--name", default=None, help="표시할 카드 이름 override")
    parser.add_argument("--output", default=None, help="저장도 같이 하고 싶을 때 출력 파일 경로 지정")
    args = parser.parse_args()

    engine = EnhancementEngine.load(args.enhancements)
    cards = load_cards(args.cards)
    card = pick_card(cards, args.card_id, args.card_type)

    image = card.render_visual(
        engine=engine,
        image_path=args.image_path,
        card_name=args.name,
    )
    image.show()
    print("shown: card preview opened")

    if args.output:
        out = card.visualize(
            engine=engine,
            output_path=args.output,
            image_path=args.image_path,
            card_name=args.name,
        )
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
