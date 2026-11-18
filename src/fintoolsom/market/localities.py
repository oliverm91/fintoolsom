from enum import Enum


class Locality(Enum):
    """Calendar code of a currency's market: an ISO country code, or the code of a
    financial-market calendar of the holidays library (EUR: TARGET2, ``"XECB"``)."""

    US = "US"
    EU = "XECB"  # TARGET2, not one euro country's public holidays
    GB = "GB"
    JP = "JP"
    CH = "CH"
    CL = "CL"
    BR = "BR"
    RU = "RU"
    CN = "CN"
    IN = "IN"
    AU = "AU"
    NZ = "NZ"
    HK = "HK"
    SG = "SG"
    KR = "KR"
    TH = "TH"
    CO = "CO"
    MX = "MX"
    BN = "BN"
    SE = "SE"
    NO = "NO"
    PE = "PE"
