import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class DataCenterOptimizer:
    def __init__(self):
        """Initialize models and scalers"""
        # Initialize models
        self.energy_model = LinearRegression()
        self.energy_scaler = StandardScaler()
        
        self.workload_classifier = KMeans(n_clusters=3, random_state=42)
        self.workload_scaler = StandardScaler()
        
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.anomaly_scaler = StandardScaler()
        
        self.is_trained = False
    
    def generate_training_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic training data"""
        np.random.seed(42)
        
        # Generate base metrics
        data = pd.DataFrame({
            'cpu_usage': np.random.uniform(0, 100, n_samples),
            'memory_usage': np.random.uniform(0, 100, n_samples),
            'network_traffic': np.random.uniform(0, 1000, n_samples),
            'disk_io': np.random.uniform(0, 500, n_samples),
            'power_consumption': np.random.uniform(50, 500, n_samples)
        })
        
        # Generate energy efficiency with realistic relationships
        data['energy_efficiency'] = (
            -0.3 * data['cpu_usage'] +
            -0.2 * data['memory_usage'] +
            -0.1 * data['network_traffic']/10 +
            -0.2 * data['power_consumption']/5 +
            np.random.normal(0, 5, n_samples) +
            100
        ).clip(0, 100)
        
        return data

    def train_models(self) -> None:
        """Train all models with generated data"""
        # Generate training data
        data = self.generate_training_data()
        
        # Prepare features for training
        features = ['cpu_usage', 'memory_usage', 'network_traffic', 'disk_io', 'power_consumption']
        X = data[features]
        y = data['energy_efficiency']
        
        # Train energy efficiency model
        X_scaled = self.energy_scaler.fit_transform(X)
        self.energy_model.fit(X_scaled, y)
        
        # Train workload classifier
        self.workload_scaler.fit_transform(X)
        self.workload_classifier.fit(self.workload_scaler.transform(X))
        
        # Train anomaly detector
        self.anomaly_scaler.fit_transform(X)
        self.anomaly_detector.fit(self.anomaly_scaler.transform(X))
        
        self.is_trained = True

    def analyze_vm(self, metrics: Dict) -> Dict:
        """Analyze VM metrics and provide recommendations"""
        if not self.is_trained:
            self.train_models()
        
        # Prepare input data
        features = ['cpu_usage', 'memory_usage', 'network_traffic', 'disk_io', 'power_consumption']
        input_df = pd.DataFrame([metrics])[features]
        
        # Get predictions from all models
        energy_input_scaled = self.energy_scaler.transform(input_df)
        predicted_efficiency = self.energy_model.predict(energy_input_scaled)[0]
        
        workload_input_scaled = self.workload_scaler.transform(input_df)
        cluster = self.workload_classifier.predict(workload_input_scaled)[0]
        
        anomaly_input_scaled = self.anomaly_scaler.transform(input_df)
        is_anomaly = self.anomaly_detector.predict(anomaly_input_scaled)[0] == -1
        anomaly_score = self.anomaly_detector.score_samples(anomaly_input_scaled)[0]
        
        # Map cluster to workload type
        workload_types = {0: "Steady", 1: "Bursty", 2: "Latency-sensitive"}
        workload_type = workload_types[cluster]
        
        # Generate recommendations
        recommendations = self.get_optimization_recommendations(
            metrics, predicted_efficiency, workload_type, is_anomaly
        )
        
        return {
            'energy_efficiency': predicted_efficiency,
            'workload_type': workload_type,
            'is_anomaly': is_anomaly,
            **recommendations
        }

    def get_optimization_recommendations(self, metrics: Dict, predicted_efficiency: float,
                                      workload_type: str, is_anomaly: bool) -> Dict:
        """Generate detailed optimization recommendations"""
        recommendations = {
            'placement_recommendation': {},
            'additional_notes': [],
            'optimization_suggestions': []
        }
        
        # Placement recommendations based on workload type
        if workload_type == "Steady":
            recommendations['placement_recommendation'] = {
                'primary': 'on-premise',
                'reasoning': 'Steady workload pattern ideal for dedicated infrastructure',
                'potential_cost_savings': '15-20%'
            }
        elif workload_type == "Bursty":
            recommendations['placement_recommendation'] = {
                'primary': 'cloud',
                'reasoning': 'Variable resource needs better served by elastic cloud resources',
                'potential_cost_savings': '25-30%'
            }
        else:  # Latency-sensitive
            recommendations['placement_recommendation'] = {
                'primary': 'hybrid',
                'reasoning': 'Balance between performance and cost optimization',
                'potential_cost_savings': '10-15%'
            }
        
        # Additional notes based on metrics
        if metrics['cpu_usage'] > 70:
            recommendations['additional_notes'].append(
                "High CPU usage suggests need for vertical scaling"
            )
        
        if metrics['memory_usage'] > 80:
            recommendations['additional_notes'].append(
                "High memory usage indicates potential memory constraints"
            )
        
        # Optimization suggestions
        if metrics['memory_usage'] > 60 and metrics['cpu_usage'] < 50:
            recommendations['optimization_suggestions'].append({
                'type': 'resource_optimization',
                'action': 'Consider reducing memory allocation',
                'potential_savings': '15-20%',
                'priority': 'Medium'
            })
        
        if predicted_efficiency < 75:
            recommendations['optimization_suggestions'].append({
                'type': 'energy_optimization',
                'action': 'Optimize workload scheduling',
                'potential_savings': '10-15%',
                'priority': 'High'
            })
        
        if is_anomaly:
            recommendations['optimization_suggestions'].append({
                'type': 'performance_optimization',
                'action': 'Investigate unusual resource usage patterns',
                'potential_savings': '5-10%',
                'priority': 'High'
            })
        
        return recommendations