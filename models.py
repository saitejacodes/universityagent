from pydantic import BaseModel
from datetime import date
from typing import Optional


class FieldMeta(BaseModel):
    """Metadata stored alongside every extracted field."""
    confidence: float            # 0.0 to 1.0 — how sure we are
    source_url: Optional[str]    # exact page this came from
    needs_review: bool = False   # flagged if confidence < 0.7
    notes: Optional[str] = None  # conflicts, caveats, flags


class AboutUniversity(BaseModel):
    name: str
    founding_year: Optional[int] = None
    ranking_qs: Optional[int] = None
    ranking_times: Optional[int] = None
    location_city: str
    location_country: str
    institution_type: Optional[str] = None  # "public" or "private"
    meta: FieldMeta


class TuitionFee(BaseModel):
    programme_level: str   # "undergraduate" | "postgraduate"
    student_type: str      # "domestic" | "international"
    annual_fee: Optional[float] = None
    currency: str
    meta: FieldMeta


class LivingCost(BaseModel):
    city: str
    monthly_rent_min: Optional[float] = None
    monthly_rent_max: Optional[float] = None
    monthly_food: Optional[float] = None
    monthly_transport: Optional[float] = None
    monthly_total_estimate: Optional[float] = None
    currency: str
    meta: FieldMeta


class Scholarship(BaseModel):
    name: str
    value: Optional[float] = None
    value_type: Optional[str] = None   # "full_tuition" | "partial" | "stipend"
    currency: Optional[str] = None
    eligibility: Optional[str] = None
    application_deadline: Optional[str] = None  # ISO date string
    renewable: Optional[bool] = None
    meta: FieldMeta


class AcceptanceRate(BaseModel):
    overall_percent: Optional[float] = None
    undergraduate_percent: Optional[float] = None
    postgraduate_percent: Optional[float] = None
    year_of_data: Optional[int] = None
    meta: FieldMeta


class GraduateEmployment(BaseModel):
    employed_within_6_months_percent: Optional[float] = None
    data_source: Optional[str] = None
    year_of_data: Optional[int] = None
    meta: FieldMeta


class AverageSalary(BaseModel):
    field_of_study: str
    median_salary: Optional[float] = None
    currency: str
    year_of_data: Optional[int] = None
    meta: FieldMeta


class VisaPolicy(BaseModel):
    country: str
    visa_type: Optional[str] = None
    processing_time_weeks_min: Optional[int] = None
    processing_time_weeks_max: Optional[int] = None
    key_requirements: list[str] = []
    application_url: Optional[str] = None
    meta: FieldMeta


class IntakeDeadline(BaseModel):
    intake_name: str             # e.g. "Fall 2025"
    application_open: Optional[str] = None   # ISO date string
    application_close: Optional[str] = None  # ISO date string
    programme: Optional[str] = None
    meta: FieldMeta


class CourseListing(BaseModel):
    code: Optional[str] = None
    title: str
    credits: Optional[int] = None
    description: Optional[str] = None
    prerequisites: list[str] = []
    mode: Optional[str] = None   # "online" | "in-person" | "hybrid"
    level: Optional[str] = None  # "undergraduate" | "postgraduate"
    meta: FieldMeta


class UniversityData(BaseModel):
    university_id: str
    university_name: str
    about: Optional[AboutUniversity] = None
    tuition_fees: list[TuitionFee] = []
    living_costs: Optional[LivingCost] = None
    scholarships: list[Scholarship] = []
    acceptance_rate: Optional[AcceptanceRate] = None
    graduate_employment: Optional[GraduateEmployment] = None
    average_salaries: list[AverageSalary] = []
    visa_policy: Optional[VisaPolicy] = None
    intake_deadlines: list[IntakeDeadline] = []
    courses: list[CourseListing] = []
    scrape_timestamp: str
    overall_confidence: float = 0.0

