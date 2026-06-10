"""
email_notifier.py — Envío de notificaciones por correo electrónico.

Usa SMTP con STARTTLS (compatible con Outlook, Office 365, Gmail, etc.).
Se activa automáticamente al finalizar la indexación de un repositorio.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import (
    EMAIL_HOST,
    EMAIL_PORT,
    EMAIL_USERNAME,
    EMAIL_PASSWORD,
    EMAIL_DESTINATARIO,
)

log = logging.getLogger(__name__)


def send_index_notification(repo: str, branch: str, success: bool, details: str = "") -> None:
    """
    Envía un correo notificando el resultado de una indexación.

    Args:
        repo: Nombre completo del repositorio (org/repo).
        branch: Rama indexada.
        success: True si la indexación terminó con éxito, False si falló.
        details: Texto adicional (estadísticas o mensaje de error).
    """
    if not all([EMAIL_HOST, EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_DESTINATARIO]):
        log.debug("Configuración de email incompleta, se omite notificación")
        return

    try:
        subject = (
            f"{'✅' if success else '❌'} Indexación "
            f"{'completada' if success else 'fallida'}: {repo} @ {branch}"
        )
        body_lines = [
            f"Repositorio: {repo}",
            f"Rama: {branch}",
            f"Estado: {'ÉXITO' if success else 'FALLO'}",
            "",
            details,
        ]
        body = "\n".join(body_lines)

        msg = MIMEMultipart()
        msg["From"] = EMAIL_USERNAME
        msg["To"] = EMAIL_DESTINATARIO
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USERNAME, EMAIL_DESTINATARIO, msg.as_string())

        log.info("Notificación de indexación enviada a %s", EMAIL_DESTINATARIO)
    except Exception as exc:
        log.error("Error enviando notificación por email: %s", exc)
