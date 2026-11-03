from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from langchain.tools import tool
from rag.chain import get_qa_chain
from config import API_URL

@tool
def get_current_philippine_date() -> dict:
    """Get today's date, weekday, and current time in the Philippines."""
    now = datetime.now(ZoneInfo("Asia/Manila"))
    return {
        "date": now.strftime("%Y-%m-%d"),
        "dayOfWeek": now.strftime("%A"),
        "time": now.strftime("%I:%M %p"),
        "timeZone": "Asia/Manila",
    }

@tool
def ask_question(question: str) -> str:
    """Search for knowledge-based answers related Bagumbayan Health Center, healthcare website and FAQs"""
    try:
        result = get_qa_chain().invoke({"query": question})
        sources = result.get("source_documents", [])

        if not sources:
            return result["result"]

        source_lines = []
        for index, source in enumerate(sources, start=1):
            matched_question = source.metadata.get("question", "Unknown source")
            score = source.metadata.get("relevance_score")
            score_text = f" (score: {score})" if score is not None else ""
            source_lines.append(f"{index}. {matched_question}{score_text}")

        return (
            f"{result['result']}\n\n"
            "Sources used:\n"
            + "\n".join(source_lines)
        )
    except Exception as e:
        return f"I don't have that information. Retrieval error: {str(e)}"

@tool
def get_services(day_of_week: str = "") -> dict:
    """Get health-center services, optionally filtered by weekday.

    The weekday must be a full name such as Thursday.
    """
    normalized_day = day_of_week.strip().capitalize()
    valid_days = {
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    }
    if normalized_day and normalized_day not in valid_days:
        return {
            "success": False,
            "message": "Provide a full weekday name, such as Thursday.",
        }

    if not API_URL:
        return {
            "success": False,
            "message": "Service lookup is not configured.",
        }

    try:
        response = httpx.get(
            f"{API_URL}/api/services",
            params={"dayOfWeek": normalized_day} if normalized_day else None,
            timeout=10,
        )

        response.raise_for_status()

        services = response.json().get("services", [])
        return {
            "success": True,
            "dayOfWeek": normalized_day or None,
            "services": [
                {
                    "serviceName": service.get("serviceName"),
                    "dayOfWeek": service.get("dayOfWeek"),
                    "startTime": service.get("startTime"),
                    "endTime": service.get("endTime"),
                }
                for service in services
            ],
        }

    except httpx.HTTPError as e:
        return {
            "success": False,
            "message": f"Failed to retrieve services: {str(e)}"
        }

@tool
def get_available_appointment_slots(
    service_name: str,
    appointment_date: str,
    period: str = "all",
    doctor_id: int = 0,
) -> dict:
    """Find doctors with available slots, then list times for a selected doctor.

    Provide service_name and appointment_date (YYYY-MM-DD). On the first
    lookup, leave doctor_id as 0 to get available doctor names and counts
    without times. After the user selects a doctor, call again with that
    doctor's doctor_id to get their available times. Period may be all,
    morning, or afternoon.
    """
    normalized_period = period.strip().lower()
    if normalized_period not in {"all", "morning", "afternoon"}:
        return {
            "success": False,
            "message": "Period must be all, morning, or afternoon.",
        }

    if not API_URL:
        return {
            "success": False,
            "message": "Appointment availability lookup is not configured.",
        }

    try:
        response = httpx.post(
            f"{API_URL}/api/ai-tools/available-slots",
            json={
                "serviceName": service_name,
                "appointmentDate": appointment_date,
                "period": normalized_period,
                **({"doctorId": doctor_id} if doctor_id > 0 else {}),
            },
            timeout=10,
        )
        response.raise_for_status()
        availability = response.json()

        return {
            "success": True,
            "serviceName": availability.get("serviceName", service_name),
            "appointmentDate": appointment_date,
            "period": normalized_period,
            "serviceExists": availability.get("serviceExists", False),
            "serviceAvailable": availability.get("serviceAvailable", False),
            "dayOfWeek": availability.get("dayOfWeek"),
            "availableServiceDays": availability.get("availableServiceDays", []),
            "matchingServiceNames": availability.get("matchingServiceNames", []),
            "selectedDoctorAvailable": availability.get(
                "selectedDoctorAvailable"
            ),
            "doctors": availability.get("doctors", []),
        }
    except httpx.HTTPError:
        return {
            "success": False,
            "message": "Could not retrieve appointment availability right now.",
        }

@tool
def get_appointment_details(reference_number: str) -> dict:
    """Get appointment details and patient information using an appointment reference number."""

    try:
        response = httpx.get(
            f"{API_URL}/api/appointments/{reference_number}",
            timeout=10,
        )

        if response.status_code == 404:
            return {
                "success": False,
                "message": f"No appointment found with reference number {reference_number}.",
            }

        response.raise_for_status()

        result = response.json()
        appointment = result.get("appointment")

        if not appointment:
            return {
                "success": False,
                "message": "Appointment information was not found.",
            }

        doctor = appointment.get("doctor") or {}
        service = appointment.get("service") or {}
        patient = appointment.get("patient") or {}
        appointment_record = appointment.get("appointmentRecord") or {}

        return {
            "success": True,
            "appointment": {
                "referenceNumber": appointment.get("referenceNumber"),
                "appointmentDate": appointment.get("appointmentDate"),
                "appointmentTime": appointment.get("appointmentTime"),
                "status": appointment.get("status"),
                "purposeOfVisit": appointment.get("purposeOfVisit"),

                "service": {
                    "serviceName": service.get("serviceName"),
                },

                "doctor": {
                    "firstName": doctor.get("firstname"),
                    "lastName": doctor.get("lastname"),
                },

                "booked by": {
                    "firstName": patient.get("firstname"),
                    "lastName": patient.get("lastname"),
                },

                "patient": {
                    "firstName": appointment_record.get("firstName"),
                    "middleName": appointment_record.get("middleName"),
                    "lastName": appointment_record.get("lastName"),
                    "suffix": appointment_record.get("suffix"),
                    "birthDate": appointment_record.get("birthDate"),
                    "gender": appointment_record.get("gender"),
                    "civilStatus": appointment_record.get("civilStatus"),
                    "contactNumber": appointment_record.get("contactNumber"),
                    "email": appointment_record.get("email"),
                    "completeAddress": appointment_record.get("completeAddress"),
                    "emergencyContactPerson": appointment_record.get(
                        "emergencyContactPerson"
                    ),
                    "emergencyContactNumber": appointment_record.get(
                        "emergencyContactNumber"
                    ),
                },
            },
        }

    except httpx.HTTPError as e:
        return {
            "success": False,
            "message": f"Failed to retrieve appointment details: {str(e)}",
        }


def getChatbotTools():
    return [
        get_current_philippine_date,
        ask_question,
        get_services,
        get_available_appointment_slots,
        get_appointment_details,
    ]
