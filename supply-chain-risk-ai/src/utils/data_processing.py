# src/utils/data_processing.py
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import pandas as pd
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
import numpy as np

class TeslaFinancialMetrics(BaseModel):
    current_price: float
    price_change_30d: float
    volume_avg_30d: float
    market_cap: float
    pe_ratio: float
    beta: float
    last_updated: str

class ProductionMetric(BaseModel):
    quarter: str = Field(alias="Quarter")
    production: int = Field(alias="Production")
    deliveries: int = Field(alias="Deliveries")
    production_gap: int = Field(alias="Production_Gap")

class SupplierMetric(BaseModel):
    ticker: str
    category: str
    current_price: float
    price_change_90d: float
    volatility_90d: float
    market_cap: float
    sector: str
    country: str
    risk_score: int
    risk_level: str
    last_updated: str

class RiskScenario(BaseModel):
    name: str
    probability: float
    trigger_events: List[str]
    financial_impact_estimate: Dict[str, float]
    mitigation_strategies: List[str]

class Alert(BaseModel):
    type: str
    severity: str
    message: str
    action: Optional[str] = None
    probability: Optional[str] = None
    score: Optional[str] = None

@dataclass
class ProcessedRiskData:
    tesla_metrics: TeslaFinancialMetrics
    production_data: pd.DataFrame
    supplier_risks: pd.DataFrame
    high_risk_scenarios: List[RiskScenario]
    active_alerts: List[Alert]
    risk_summary: Dict[str, float]

class TeslaRiskDataProcessor:
    """
    Processes Tesla supply chain risk data from JSON format
    into structured, analysis-ready formats
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'low': (0, 3),
            'medium': (3, 6),  
            'high': (6, 8),
            'critical': (8, 10)
        }
    
    def process_tesla_data(self, raw_data: Dict) -> ProcessedRiskData:
        """
        Main processing function that transforms raw Tesla data
        """
        tesla_data = raw_data['tesla_comprehensive_risk_analysis']
        
        # Process core Tesla metrics
        tesla_metrics = self._process_tesla_metrics(tesla_data['tesla_core_metrics'])
        
        # Process production data
        production_df = self._process_production_data(tesla_data['tesla_core_metrics']['production_metrics'])
        
        # Process supplier network
        supplier_df = self._process_supplier_network(tesla_data['supplier_network'])
        
        # Process risk scenarios
        scenarios = self._process_scenarios(tesla_data['scenario_modeling'])
        
        # Process alerts
        alerts = self._process_alerts(tesla_data['alert_system'])
        
        # Generate risk summary
        risk_summary = self._generate_risk_summary(tesla_data['comprehensive_risk_assessment'])
        
        return ProcessedRiskData(
            tesla_metrics=tesla_metrics,
            production_data=production_df,
            supplier_risks=supplier_df,
            high_risk_scenarios=scenarios,
            active_alerts=alerts,
            risk_summary=risk_summary
        )
    
    def _process_tesla_metrics(self, core_metrics: Dict) -> TeslaFinancialMetrics:
        """Process Tesla's core financial metrics"""
        financial_data = core_metrics['financial']
        return TeslaFinancialMetrics(**financial_data)
    
    def _process_production_data(self, production_metrics: List[Dict]) -> pd.DataFrame:
        """Convert production metrics to DataFrame with additional analysis"""
        df = pd.DataFrame(production_metrics)
        
        # Add calculated fields
        df['delivery_efficiency'] = (df['Deliveries'] / df['Production']) * 100
        df['gap_percentage'] = (df['Production_Gap'] / df['Production']) * 100
        df['quarter_num'] = df['Quarter'].str.extract(r'Q(\d)').astype(int)
        df['year'] = df['Quarter'].str.extract(r'(\d{4})').astype(int)
        
        # Add risk flags
        df['high_gap_risk'] = df['Production_Gap'].abs() > 30000
        df['efficiency_risk'] = df['delivery_efficiency'] < 95
        
        return df
    
    def _process_supplier_network(self, supplier_data: Dict) -> pd.DataFrame:
        """Process supplier network into analysis-ready format"""
        suppliers = []
        
        # Process different supplier categories
        categories = ['lithium_suppliers', 'metals_suppliers']
        
        for category in categories:
            if category in supplier_data:
                for supplier_name, supplier_info in supplier_data[category].items():
                    supplier_record = {
                        'supplier_name': supplier_name,
                        'category': supplier_info.get('category', category.split('_')[0]),
                        **supplier_info
                    }
                    suppliers.append(supplier_record)
        
        df = pd.DataFrame(suppliers)
        
        if not df.empty:
            # Add risk categorization
            df['risk_category'] = df['risk_score'].apply(self._categorize_risk)
            df['volatility_risk'] = df['volatility_90d'] > df['volatility_90d'].quantile(0.75)
            df['price_decline_risk'] = df['price_change_90d'] < -15
            df['combined_risk_score'] = self._calculate_combined_risk(df)
        
        return df
    
    def _process_scenarios(self, scenario_data: Dict) -> List[RiskScenario]:
        """Extract high-probability risk scenarios"""
        high_risk_scenarios = []
        
        for scenario_name, scenario_info in scenario_data.items():
            if scenario_info.get('probability', 0) > 0.1:  # 10% threshold
                scenario = RiskScenario(
                    name=scenario_info['name'],
                    probability=scenario_info['probability'],
                    trigger_events=scenario_info.get('trigger_events', []),
                    financial_impact_estimate=scenario_info.get('financial_impact_estimate', {}),
                    mitigation_strategies=scenario_info.get('mitigation_strategies', [])
                )
                high_risk_scenarios.append(scenario)
        
        # Sort by probability (highest first)
        high_risk_scenarios.sort(key=lambda x: x.probability, reverse=True)
        
        return high_risk_scenarios
    
    def _process_alerts(self, alert_data: Dict) -> List[Alert]:
        """Process active alerts with priority sorting"""
        all_alerts = []
        
        # Process different alert categories
        for alert_category, alerts in alert_data.items():
            for alert_info in alerts:
                alert = Alert(**alert_info)
                all_alerts.append(alert)
        
        # Sort by severity (Critical > High > Medium > Low)
        severity_order = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        all_alerts.sort(key=lambda x: severity_order.get(x.severity, 0), reverse=True)
        
        return all_alerts
    
    def _generate_risk_summary(self, risk_assessment: Dict) -> Dict[str, float]:
        """Generate comprehensive risk summary metrics"""
        enhanced_scores = risk_assessment.get('enhanced_risk_scores', {})
        
        risk_summary = {
            'overall_risk': risk_assessment['overall_risk_metrics']['overall_risk_score'],
            'tesla_risk': enhanced_scores.get('tesla_overall', {}).get('final_score', 0),
            'supplier_risk': enhanced_scores.get('supplier_network', {}).get('final_score', 0),
            'production_risk': enhanced_scores.get('production_capacity', {}).get('final_score', 0)
        }
        
        # Add weighted composite score
        weights = {'tesla_risk': 0.3, 'supplier_risk': 0.4, 'production_risk': 0.3}
        risk_summary['composite_score'] = sum(
            risk_summary[key] * weight for key, weight in weights.items()
        )
        
        return risk_summary
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into text labels"""
        for category, (min_val, max_val) in self.risk_thresholds.items():
            if min_val <= risk_score < max_val:
                return category.title()
        return 'Critical'
    
    def _calculate_combined_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate combined risk score for suppliers"""
        if df.empty:
            return pd.Series()
        
        # Normalize different risk factors
        volatility_norm = df['volatility_90d'] / df['volatility_90d'].max()
        price_decline_norm = np.abs(df['price_change_90d'].clip(upper=0)) / 100
        base_risk_norm = df['risk_score'] / 10
        
        # Weighted combination
        combined = (base_risk_norm * 0.4 + 
                   volatility_norm * 0.3 + 
                   price_decline_norm * 0.3) * 10
        
        return combined.round(2)
    
    def generate_insights(self, processed_data: ProcessedRiskData) -> Dict[str, str]:
        """Generate actionable insights from processed data"""
        insights = {}
        
        # Production insights
        recent_production = processed_data.production_data.iloc[-1]
        if abs(recent_production['Production_Gap']) > 30000:
            insights['production'] = f"High production gap of {recent_production['Production_Gap']:,} vehicles in {recent_production['Quarter']}"
        
        # Supplier insights
        critical_suppliers = processed_data.supplier_risks[
            processed_data.supplier_risks['risk_category'] == 'Critical'
        ]
        if not critical_suppliers.empty:
            insights['suppliers'] = f"{len(critical_suppliers)} suppliers in critical risk category"
        
        # Scenario insights
        high_prob_scenarios = [s for s in processed_data.high_risk_scenarios if s.probability > 0.2]
        if high_prob_scenarios:
            top_scenario = high_prob_scenarios[0]
            insights['scenarios'] = f"Highest risk scenario: {top_scenario.name} ({top_scenario.probability:.1%} probability)"
        
        # Alert insights
        critical_alerts = [a for a in processed_data.active_alerts if a.severity == 'Critical']
        if critical_alerts:
            insights['alerts'] = f"{len(critical_alerts)} critical alerts requiring immediate attention"
        
        return insights

# Usage example and utility functions
def load_and_process_tesla_data(file_path: str) -> ProcessedRiskData:
    """Convenience function to load and process Tesla data from JSON file"""
    with open(file_path, 'r') as f:
        raw_data = json.load(f)
    
    processor = TeslaRiskDataProcessor()
    return processor.process_tesla_data(raw_data)

def export_processed_data(processed_data: ProcessedRiskData, output_dir: str):
    """Export processed data to CSV files for further analysis"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Export production data
    processed_data.production_data.to_csv(
        os.path.join(output_dir, 'tesla_production_analysis.csv'), 
        index=False
    )
    
    # Export supplier risks
    processed_data.supplier_risks.to_csv(
        os.path.join(output_dir, 'supplier_risk_analysis.csv'), 
        index=False
    )
    
    # Export risk summary
    with open(os.path.join(output_dir, 'risk_summary.json'), 'w') as f:
        json.dump(processed_data.risk_summary, f, indent=2)

if __name__ == "__main__":
    # Example usage
    processor = TeslaRiskDataProcessor()
    
    # Load your data (replace with actual file path)
    # processed_data = load_and_process_tesla_data('data/raw/tesla_risk_data.json')
    
    # Generate insights
    # insights = processor.generate_insights(processed_data)
    # print("Key Insights:", insights)
    
    print("Tesla Risk Data Processor ready for use!")