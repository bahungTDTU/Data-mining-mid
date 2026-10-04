from typing import Annotated, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Credentials(Input):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


PositiveID = Annotated[int, Field(gt=0, strict=True)]


class Preferences(Input):
    game_ids: list[PositiveID] = Field(max_length=100)
    tag_ids: list[PositiveID] = Field(default_factory=list, max_length=50)

    @field_validator('game_ids', 'tag_ids')
    @classmethod
    def unique(cls, values):
        if len(values) != len(set(values)):
            raise ValueError('IDs must be unique')
        return values


class GameInput(Input):
    game_id: PositiveID


class Feedback(GameInput):
    type: Literal['like', 'dislike', 'click']
    idempotency_key: UUID
