from abc import ABCMeta, abstractmethod
from typing import Optional
from mailjet_rest import Client


class EmailClient(metaclass=ABCMeta):

    def __init__(self, api_key: str, secret_key: str):
        self._api_key = api_key
        self._secret_key = secret_key

    @abstractmethod
    def send_email(
        self,
        from_email: str,
        to_email: str,
        subject: str,
        content: str,
        from_name: Optional[str],
        to_name: Optional[str],
    ) -> bool:
        pass


class MailjetEmailClient(EmailClient):
    
    def __init__(self, api_key: str, secret_key: str):
        super().__init__(api_key, secret_key)
        self._client = Client(
            auth=(self._api_key, self._secret_key),
            version="v3.1"
        )

    @property
    def client(self) -> Client:
        return self._client

    def send_email(
        self,
        from_email: str,
        to_email: str,
        subject: str,
        content: str,
        from_name: Optional[str] = None,
        to_name: Optional[str] = None,
    ) -> bool:
        data = {
            "Messages": [{
                "From": {
                    "Email": from_email,
                    "Name": from_name,
                },
                "To": [
                    {
                        "Email": to_email,
                        "Name": to_name,
                    }
                ],
                "Subject": subject,
                "HTMLPart": content,
            }]
        }
        try:
            result = self.client.send.create(data=data)
            return True  # Assuming that no exception means success
        except Exception as ex:
            print(f"Failed to send email: {ex}")
            return False
