# Localities have currencies and calendars

from dataclasses import dataclass, field

from .currencies import Currency
from ..dates import Calendar

@dataclass(slots=True)
class Locality:
    name: str
    currency: Currency
    calendar: Calendar = field(default=None)

    def __post_init__(self):
        self.name = self.name.upper()
        if self.calendar is None:
            try:
                self.calendar = Calendar(country=self.name)
            except NotImplementedError:
                print(f'Warning. No calendar for {self.name}. Using default calendar (only holidays are Saturday and Sunday).')
                self.calendar = Calendar()