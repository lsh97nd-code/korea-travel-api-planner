import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import requests
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
)
from pydantic import BaseModel


DATE_FORMAT = "%Y-%m-%d"
DATE_FORMAT_LABEL = "YYYY-MM-DD"


class MissingAPIKeyError(Exception):
    """필요한 API 키가 환경변수에 설정되지 않았을 때 발생합니다."""
    pass


class TravelRecommendation(BaseModel):
    """하나의 여행지 추천 정보입니다."""

    recommended_city: str
    weather: str
    events: list[str]
    reason: str


class TravelRecommendations(BaseModel):
    """복수의 여행지 추천 정보를 담는 구조입니다."""

    recommendations: list[TravelRecommendation]


class Restaurant(TypedDict):
    """Kakao Local API에서 사용할 맛집 정보입니다."""

    place_name: str
    address: str
    category: str
    url: str


def load_api_keys() -> tuple[str, str]:
    """환경변수에서 OpenAI API 키와 Kakao REST API 키를 불러옵니다."""

    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    kakao_rest_api_key = os.getenv("KAKAO_REST_API_KEY")

    missing_keys = []

    if not openai_api_key:
        missing_keys.append("OPENAI_API_KEY")

    if not kakao_rest_api_key:
        missing_keys.append("KAKAO_REST_API_KEY")

    if missing_keys:
        raise MissingAPIKeyError(
            f"누락된 환경변수: {', '.join(missing_keys)}\n"
            ".env 파일에 API 키를 설정한 후 다시 실행하세요."
        )

    return openai_api_key, kakao_rest_api_key


def parse_date(date_text: str) -> datetime:
    """여행 날짜를 YYYY-MM-DD 형식으로 검사하고 datetime 객체로 변환합니다."""

    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text):
        raise argparse.ArgumentTypeError(
            f'"{date_text}"는 올바른 날짜 형식이 아닙니다. '
            f'"{DATE_FORMAT_LABEL}" 형식으로 입력하세요.'
        )

    try:
        return datetime.strptime(date_text, DATE_FORMAT)

    except ValueError:
        raise argparse.ArgumentTypeError(
            f'"{date_text}"는 존재하지 않는 날짜입니다. '
            f'"{DATE_FORMAT_LABEL}" 형식의 올바른 날짜를 입력하세요.'
        )


def parse_args() -> argparse.Namespace:
    """명령행 인수를 처리합니다."""

    parser = argparse.ArgumentParser(
        description="API를 활용한 국내 여행지 추천 프로그램"
    )

    parser.add_argument(
        "-date",
        "-d",
        "--date",
        dest="date",
        required=True,
        type=parse_date,
        metavar=DATE_FORMAT_LABEL,
        help=(
            f'여행 날짜를 "{DATE_FORMAT_LABEL}" 형식으로 입력하세요. '
            "예: 2026-09-15"
        ),
    )

    return parser.parse_args()


def get_travel_recommendation(
    client: OpenAI,
    travel_date: str,
) -> TravelRecommendations:
    """OpenAI API를 호출하여 여행 날짜에 맞는 국내 여행지를 추천받습니다.

    LLM의 구조화된 결과를 읽지 못하거나 JSON 파싱에 실패한 경우
    1회만 재시도합니다.

    Raises:
        AuthenticationError: OpenAI API 키가 유효하지 않은 경우
        APIConnectionError: OpenAI 서버 연결에 실패한 경우
        APIStatusError: OpenAI API가 오류 상태 코드를 반환한 경우
        ValueError: 재시도 후에도 LLM의 구조화된 응답을 읽을 수 없는 경우
    """

    for attempt in range(2):
        try:
            response = client.responses.parse(
                model="gpt-5.6-luna",
                input=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 대한민국 국내 여행 전문가입니다. "
                            "사용자가 입력한 여행 날짜를 기준으로 "
                            "서로 다른 국내 여행지 3곳을 추천하세요. "
                            "recommended_city에는 대한민국의 시 또는 군 단위 "
                            "지역명을 작성하세요. "
                            "weather에는 해당 시기의 일반적인 날씨 특징을 "
                            "간단히 작성하세요. "
                            "events에는 해당 날짜 전후에 고려할 만한 행사나 "
                            "계절 활동을 1개 이상 3개 이하로 작성하세요. "
                            "reason에는 추천 이유를 한국어 2~4문장으로 작성하세요."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"여행 날짜: {travel_date}",
                    },
                ],
                text_format=TravelRecommendations,
            )

        except ValueError as error:
            if attempt == 0:
                print(
                    "LLM JSON 파싱 실패: "
                    "1회 재시도합니다."
                )
                continue

            raise ValueError(
                "LLM의 여행지 추천 결과를 "
                "재시도 후에도 파싱하지 못했습니다."
            ) from error

        if response.output_parsed is not None:
            return response.output_parsed

        if attempt == 0:
            print(
                "LLM 결과 처리 실패: "
                "1회 재시도합니다."
            )
            continue

    raise ValueError(
        "LLM의 여행지 추천 결과를 "
        "재시도 후에도 읽을 수 없습니다."
    )


def search_restaurants(
    kakao_rest_api_key: str,
    city: str,
) -> list[Restaurant]:
    """Kakao Local API를 사용하여 추천 지역의 맛집을 검색합니다.

    Raises:
        requests.HTTPError: API 응답 상태 코드가 4xx 또는 5xx인 경우
        requests.ConnectionError: Kakao 서버 연결에 실패한 경우
        requests.Timeout: 요청 시간이 초과된 경우
        ValueError: Kakao API 응답 구조가 올바르지 않은 경우
    """

    url = "https://dapi.kakao.com/v2/local/search/keyword.json"

    headers = {
        "Authorization": f"KakaoAK {kakao_rest_api_key}"
    }

    params = {
        "query": f"{city} 맛집",
        "category_group_code": "FD6",
        "size": 5,
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    if "documents" not in data or not isinstance(data["documents"], list):
        raise ValueError("Kakao API 응답 구조가 올바르지 않습니다.")

    restaurants: list[Restaurant] = []

    for place in data["documents"]:
        restaurants.append(
            {
                "place_name": place.get("place_name", ""),
                "address": (
                    place.get("road_address_name")
                    or place.get("address_name", "")
                ),
                "category": place.get("category_name", ""),
                "url": place.get("place_url", ""),
            }
        )

    return restaurants


def generate_travel_guide(
    client: OpenAI,
    travel_date: str,
    recommendations: TravelRecommendations,
    restaurants_by_city: dict[str, list[Restaurant]],
) -> str:
    """복수 여행지 추천 정보와 지역별 맛집 정보를 이용하여 최종 여행 안내서를 생성합니다.

    Raises:
        AuthenticationError: OpenAI API 키가 유효하지 않은 경우
        APIConnectionError: OpenAI 서버 연결에 실패한 경우
        APIStatusError: OpenAI API가 오류 상태 코드를 반환한 경우
        ValueError: 최종 여행 안내서가 생성되지 않은 경우
    """

    recommendation_sections = []

    for index, recommendation in enumerate(
        recommendations.recommendations,
        start=1,
    ):
        restaurants = restaurants_by_city.get(
            recommendation.recommended_city,
            [],
        )

        if restaurants:
            restaurant_text = "\n".join(
                (
                    f"- {restaurant['place_name']} | "
                    f"주소: {restaurant['address']} | "
                    f"분류: {restaurant['category']} | "
                    f"URL: {restaurant['url']}"
                )
                for restaurant in restaurants
            )
        else:
            restaurant_text = "데이터 없음"

        section = (
            f"[추천 지역 {index}]\n"
            f"추천 지역: {recommendation.recommended_city}\n"
            f"날씨: {recommendation.weather}\n"
            f"행사/활동: {', '.join(recommendation.events)}\n"
            f"추천 이유: {recommendation.reason}\n"
            f"맛집 정보:\n{restaurant_text}"
        )

        recommendation_sections.append(section)

    recommendation_text = (
        f"여행 날짜: {travel_date}\n\n"
        + "\n\n".join(recommendation_sections)
    )

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=[
            {
                "role": "system",
                "content": (
                    "당신은 대한민국 국내 여행 일정 전문가입니다. "
                    "제공된 복수의 여행지 추천 정보와 각 지역의 맛집 정보를 바탕으로 "
                    "실용적인 국내 여행 안내서를 Markdown 형식으로 작성하세요. "
                    "각 추천 지역을 구분하여 작성하세요. "
                    "각 지역마다 반드시 추천 지역, 추천 이유, 날씨, 행사/활동, 맛집, "
                    "오전 일정, 오후 일정, 저녁 일정을 포함하세요. "
                    "제공되지 않은 맛집이나 구체적인 행사 정보를 "
                    "임의로 만들어내지 마세요."
                ),
            },
            {
                "role": "user",
                "content": recommendation_text,
            },
        ],
    )

    output_text = response.output_text

    if not output_text:
        raise ValueError("최종 여행 안내서를 생성하지 못했습니다.")

    guide = output_text.strip()

    if not guide:
        raise ValueError("최종 여행 안내서를 생성하지 못했습니다.")

    return guide


def load_cached_results(
    travel_date: str,
) -> tuple[
    TravelRecommendations,
    dict[str, list[Restaurant]],
    list[str],
    str,
] | None:
    """같은 날짜의 기존 결과 파일이 있으면 캐시 데이터를 불러옵니다."""

    results_dir = Path("results")

    raw_json_path = results_dir / f"{travel_date}_raw.json"
    guide_path = results_dir / f"{travel_date}_travel_guide.md"

    if not raw_json_path.exists() or not guide_path.exists():
        return None

    try:
        with raw_json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_result = json.load(file)

        recommendations = TravelRecommendations(
            recommendations=raw_result["recommendations"]
        )

        restaurants_by_city = raw_result["restaurants_by_city"]
        errors = raw_result.get("errors", [])

        if not isinstance(restaurants_by_city, dict):
            raise ValueError(
                "캐시의 restaurants_by_city 구조가 올바르지 않습니다."
            )

        if not isinstance(errors, list):
            raise ValueError(
                "캐시의 errors 구조가 올바르지 않습니다."
            )

        with guide_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            guide = file.read()

        if not guide.strip():
            raise ValueError(
                "캐시된 여행 안내서가 비어 있습니다."
            )

    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"캐시 불러오기 실패: {error}"
        )
        print(
            "기존 캐시를 사용하지 않고 API 호출을 진행합니다."
        )
        return None

    return (
        recommendations,
        restaurants_by_city,
        errors,
        guide,
    )


def save_results(
    travel_date: str,
    recommendations: TravelRecommendations,
    restaurants_by_city: dict[str, list[Restaurant]],
    guide: str,
    errors: list[str],
) -> tuple[Path, Path]:
    """복수 여행지 추천 원본 데이터와 최종 여행 안내서를 results 폴더에 저장합니다.

    Raises:
        OSError: 파일 시스템 접근 권한이 없거나 파일 저장에 실패한 경우
    """

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_result = {
        "travel_date": travel_date,
        "recommendations": recommendations.model_dump()["recommendations"],
        "restaurants_by_city": restaurants_by_city,
        "errors": errors,
    }

    raw_json_path = results_dir / f"{travel_date}_raw.json"
    guide_path = results_dir / f"{travel_date}_travel_guide.md"

    with raw_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            raw_result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with guide_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(guide)

    return raw_json_path, guide_path


def _handle_openai_error(error: Exception) -> None:
    """OpenAI API 관련 예외 메시지를 출력하고 프로그램을 종료합니다."""

    if isinstance(error, AuthenticationError):
        print(
            "오류: OpenAI API 키가 유효하지 않습니다. "
            ".env 파일을 확인하세요."
        )

    elif isinstance(error, APIConnectionError):
        print(
            "오류: OpenAI 서버에 연결할 수 없습니다. "
            "네트워크 상태를 확인하세요."
        )

    elif isinstance(error, APIStatusError):
        print(
            "오류: OpenAI API 응답 오류 "
            f"(HTTP {error.status_code}): {error.message}"
        )

    else:
        print(f"오류: {error}")

    raise SystemExit(1)


def main() -> None:
    args = parse_args()

    try:
        openai_api_key, kakao_rest_api_key = load_api_keys()

    except MissingAPIKeyError as error:
        print(f"오류: {error}")
        raise SystemExit(1)

    travel_date = args.date.strftime(DATE_FORMAT)

    print(f"입력한 여행 날짜: {travel_date}")
    print("API 키 설정 확인: 완료")

    cached_result = load_cached_results(travel_date)

    if cached_result is not None:
        (
            recommendations,
            restaurants_by_city,
            errors,
            guide,
        ) = cached_result

        raw_json_path = (
            Path("results")
            / f"{travel_date}_raw.json"
        )

        guide_path = (
            Path("results")
            / f"{travel_date}_travel_guide.md"
        )

        print("\n[캐시 사용]")
        print(
            "기존 결과를 불러왔습니다. "
            "API를 호출하지 않습니다."
        )
        print(f"원본 데이터: {raw_json_path}")
        print(f"여행 안내서: {guide_path}")

        print("\n" + "=" * 50)
        print("[최종 여행 안내서]")
        print("=" * 50)
        print(guide)

        return

    errors: list[str] = []

    client = OpenAI(api_key=openai_api_key)

    try:
        recommendations = get_travel_recommendation(
            client,
            travel_date,
        )

    except (
        AuthenticationError,
        APIConnectionError,
        APIStatusError,
        ValueError,
    ) as error:
        _handle_openai_error(error)

    print("\n[여행지 추천 결과]")

    restaurants_by_city: dict[str, list[Restaurant]] = {}

    for city_index, recommendation in enumerate(
        recommendations.recommendations,
        start=1,
    ):
        print("\n" + "-" * 50)
        print(f"[추천 지역 {city_index}]")
        print(f"추천 지역: {recommendation.recommended_city}")
        print(f"날씨: {recommendation.weather}")
        print("행사/활동:")

        for event in recommendation.events:
            print(f"- {event}")

        print(f"추천 이유: {recommendation.reason}")

        try:
            restaurants = search_restaurants(
                kakao_rest_api_key,
                recommendation.recommended_city,
            )

        except requests.RequestException as error:
            error_message = (
                f"{recommendation.recommended_city} "
                f"Kakao 맛집 검색 오류: {error}"
            )
            print(f"\n맛집 검색 오류: {error}")
            errors.append(error_message)
            restaurants = []

        except ValueError as error:
            error_message = (
                f"{recommendation.recommended_city} "
                f"Kakao 맛집 검색 오류: {error}"
            )
            print(f"\n맛집 검색 오류: {error}")
            errors.append(error_message)
            restaurants = []

        restaurants_by_city[
            recommendation.recommended_city
        ] = restaurants

        print(
            f"\n[{recommendation.recommended_city} "
            "맛집 검색 결과]"
        )

        if not restaurants:
            print("데이터 없음")

        else:
            for restaurant_index, restaurant in enumerate(
                restaurants,
                start=1,
            ):
                print(
                    f"\n{restaurant_index}. "
                    f"{restaurant['place_name']}"
                )
                print(f"주소: {restaurant['address']}")
                print(f"분류: {restaurant['category']}")
                print(f"URL: {restaurant['url']}")

    print("\n[최종 여행 안내서 생성 중...]")

    try:
        guide = generate_travel_guide(
            client,
            travel_date,
            recommendations,
            restaurants_by_city,
        )

    except (
        AuthenticationError,
        APIConnectionError,
        APIStatusError,
        ValueError,
    ) as error:
        _handle_openai_error(error)

    print("\n" + "=" * 50)
    print("[최종 여행 안내서]")
    print("=" * 50)
    print(guide)

    try:
        raw_json_path, guide_path = save_results(
            travel_date,
            recommendations,
            restaurants_by_city,
            guide,
            errors,
        )

    except OSError as error:
        print(f"\n파일 저장 오류: {error}")

    else:
        print("\n[저장 완료]")
        print(f"원본 데이터: {raw_json_path}")
        print(f"여행 안내서: {guide_path}")


if __name__ == "__main__":
    main()