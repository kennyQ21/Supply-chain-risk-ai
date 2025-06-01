class AlertParser(BaseOutputParser):
    """Parse alert generation output"""
    
    def parse(self, text: str) -> List[Dict]:
        try:
            alerts = []
            # Parse multiple alerts from text
            alert_sections = self._split_alerts(text)
            
            for section in alert_sections:
                alert = {
                    "type": self._extract_alert_type(section),
                    "severity": self._extract_severity(section),
                    "message": self._extract_message(section),
                    "action": self._extract_action(section),
                    "timeline": self._extract_timeline(section)
                }
                alerts.append(alert)
            
            return alerts
        except Exception as e:
            return [{"error": f"Alert parsing failed: {str(e)}"}]