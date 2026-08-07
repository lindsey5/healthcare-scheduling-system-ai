import httpx
from langchain.tools import tool
from rag.chain import get_qa_chain
from config import API_URL

@tool
def ask_question(question: str) -> str:
    """Search for knowledge-based answers related Bagumbayan Health Center, healthcare website and FAQs"""
    try:
        result = get_qa_chain().invoke({"query": question})
        print(result)
        return result["result"]
    except Exception as e:
            return f"{str(e)}"

@tool
def get_services() -> dict:
    """Get the healthcare services offered by the health center"""

    try:
        response = httpx.get(
            f"{API_URL}/api/services",
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    except httpx.HTTPError as e:
        return {
            "success": False,
            "message": f"Failed to retrieve appointment details: {str(e)}"
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
    return [ask_question, get_services, get_appointment_details]