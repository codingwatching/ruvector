//! iOS Self-Learning Module
//!
//! Privacy-preserving on-device learning from iOS-specific data sources:
//! - HealthKit: Activity, sleep, heart rate patterns
//! - Location: Movement patterns, frequently visited places
//! - Communication: Call/message timing patterns (metadata only)
//! - Calendar: Schedule patterns, availability windows
//! - App Usage: Usage patterns and preferences
//!
//! All learning happens on-device with no data leaving the device.

use std::collections::HashMap;
use std::vec::Vec;

// ============================================
// Health Learning (HealthKit Integration)
// ============================================

/// Health metric types from HealthKit
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HealthMetric {
    /// Step count
    Steps = 0,
    /// Active energy burned (calories)
    ActiveEnergy = 1,
    /// Heart rate (BPM)
    HeartRate = 2,
    /// Resting heart rate
    RestingHeartRate = 3,
    /// Heart rate variability
    HeartRateVariability = 4,
    /// Sleep duration (hours)
    SleepDuration = 5,
    /// Sleep quality (0-1)
    SleepQuality = 6,
    /// Workout duration (minutes)
    WorkoutDuration = 7,
    /// Stand hours
    StandHours = 8,
    /// Exercise minutes
    ExerciseMinutes = 9,
    /// Distance walked/run (km)
    Distance = 10,
    /// Flights climbed
    FlightsClimbed = 11,
    /// Mindfulness minutes
    MindfulMinutes = 12,
    /// Respiratory rate
    RespiratoryRate = 13,
    /// Blood oxygen (SpO2)
    BloodOxygen = 14,
}

impl HealthMetric {
    /// Get typical range for normalization
    pub fn typical_range(&self) -> (f32, f32) {
        match self {
            HealthMetric::Steps => (0.0, 15000.0),
            HealthMetric::ActiveEnergy => (0.0, 1000.0),
            HealthMetric::HeartRate => (40.0, 180.0),
            HealthMetric::RestingHeartRate => (40.0, 100.0),
            HealthMetric::HeartRateVariability => (0.0, 100.0),
            HealthMetric::SleepDuration => (0.0, 12.0),
            HealthMetric::SleepQuality => (0.0, 1.0),
            HealthMetric::WorkoutDuration => (0.0, 180.0),
            HealthMetric::StandHours => (0.0, 16.0),
            HealthMetric::ExerciseMinutes => (0.0, 120.0),
            HealthMetric::Distance => (0.0, 20.0),
            HealthMetric::FlightsClimbed => (0.0, 50.0),
            HealthMetric::MindfulMinutes => (0.0, 60.0),
            HealthMetric::RespiratoryRate => (8.0, 30.0),
            HealthMetric::BloodOxygen => (90.0, 100.0),
        }
    }

    /// Normalize value to 0-1 range
    pub fn normalize(&self, value: f32) -> f32 {
        let (min, max) = self.typical_range();
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }
}

/// Health state snapshot
#[derive(Clone, Debug, Default)]
pub struct HealthState {
    /// Current metrics (normalized 0-1)
    pub metrics: HashMap<HealthMetric, f32>,
    /// Hour of day (0-23)
    pub hour: u8,
    /// Day of week (0-6, 0=Sunday)
    pub day_of_week: u8,
    /// Is workout active
    pub workout_active: bool,
}

impl HealthState {
    /// Create from raw HealthKit values
    pub fn from_healthkit(
        steps: f32,
        active_energy: f32,
        heart_rate: f32,
        sleep_hours: f32,
        hour: u8,
        day_of_week: u8,
    ) -> Self {
        let mut metrics = HashMap::new();
        metrics.insert(HealthMetric::Steps, HealthMetric::Steps.normalize(steps));
        metrics.insert(HealthMetric::ActiveEnergy, HealthMetric::ActiveEnergy.normalize(active_energy));
        metrics.insert(HealthMetric::HeartRate, HealthMetric::HeartRate.normalize(heart_rate));
        metrics.insert(HealthMetric::SleepDuration, HealthMetric::SleepDuration.normalize(sleep_hours));

        Self {
            metrics,
            hour,
            day_of_week,
            workout_active: false,
        }
    }

    /// Convert to feature vector for learning
    pub fn to_features(&self) -> Vec<f32> {
        let mut features = vec![0.0; 20];

        // Metrics (0-14)
        for i in 0..15 {
            if let Some(&val) = self.metrics.get(&unsafe { std::mem::transmute::<u8, HealthMetric>(i) }) {
                features[i as usize] = val;
            }
        }

        // Time encoding (15-17)
        features[15] = (self.hour as f32 * std::f32::consts::PI / 12.0).sin(); // Hour sin
        features[16] = (self.hour as f32 * std::f32::consts::PI / 12.0).cos(); // Hour cos
        features[17] = self.day_of_week as f32 / 6.0; // Day normalized

        // Flags (18-19)
        features[18] = if self.workout_active { 1.0 } else { 0.0 };
        features[19] = 0.0; // Reserved

        features
    }
}

/// Health pattern learner
pub struct HealthLearner {
    /// Daily patterns (hour -> average metrics)
    daily_patterns: Vec<Vec<f32>>,
    /// Weekly patterns (day -> average metrics)
    weekly_patterns: Vec<Vec<f32>>,
    /// Running averages for each metric
    metric_averages: Vec<f32>,
    /// Sample count for averaging
    sample_count: u64,
    /// Anomaly thresholds (std dev multiplier)
    anomaly_threshold: f32,
}

impl HealthLearner {
    pub fn new() -> Self {
        Self {
            daily_patterns: vec![vec![0.0; 15]; 24], // 24 hours x 15 metrics
            weekly_patterns: vec![vec![0.0; 15]; 7], // 7 days x 15 metrics
            metric_averages: vec![0.0; 15],
            sample_count: 0,
            anomaly_threshold: 2.0,
        }
    }

    /// Learn from a health state observation
    pub fn learn(&mut self, state: &HealthState) {
        let hour = (state.hour as usize) % 24;
        let day = (state.day_of_week as usize) % 7;

        // Update daily pattern
        for (metric, &value) in &state.metrics {
            let idx = *metric as usize;
            if idx < 15 {
                // Exponential moving average
                let alpha = 0.1;
                self.daily_patterns[hour][idx] =
                    (1.0 - alpha) * self.daily_patterns[hour][idx] + alpha * value;
                self.weekly_patterns[day][idx] =
                    (1.0 - alpha) * self.weekly_patterns[day][idx] + alpha * value;
                self.metric_averages[idx] =
                    (1.0 - alpha) * self.metric_averages[idx] + alpha * value;
            }
        }

        self.sample_count += 1;
    }

    /// Get expected health state for given time
    pub fn predict(&self, hour: u8, day_of_week: u8) -> Vec<f32> {
        let h = (hour as usize) % 24;
        let d = (day_of_week as usize) % 7;

        // Blend daily and weekly patterns
        let mut prediction = vec![0.0; 15];
        for i in 0..15 {
            prediction[i] = 0.7 * self.daily_patterns[h][i] + 0.3 * self.weekly_patterns[d][i];
        }
        prediction
    }

    /// Detect anomalies in current state
    pub fn detect_anomalies(&self, state: &HealthState) -> Vec<(HealthMetric, f32)> {
        let mut anomalies = Vec::new();
        let predicted = self.predict(state.hour, state.day_of_week);

        for (metric, &actual) in &state.metrics {
            let idx = *metric as usize;
            if idx < 15 {
                let expected = predicted[idx];
                let diff = (actual - expected).abs();

                if diff > self.anomaly_threshold * 0.2 {
                    // 0.2 is approximate std dev for normalized values
                    anomalies.push((*metric, diff));
                }
            }
        }

        anomalies
    }

    /// Get energy level estimation (0-1)
    pub fn estimate_energy(&self, state: &HealthState) -> f32 {
        let steps = state.metrics.get(&HealthMetric::Steps).unwrap_or(&0.5);
        let active = state.metrics.get(&HealthMetric::ActiveEnergy).unwrap_or(&0.5);
        let sleep = state.metrics.get(&HealthMetric::SleepDuration).unwrap_or(&0.5);
        let hr = state.metrics.get(&HealthMetric::HeartRate).unwrap_or(&0.5);

        // Higher energy if well-rested and active
        let rest_factor = (*sleep).min(1.0);
        let activity_factor = (*steps * 0.5 + *active * 0.5).min(1.0);
        let hr_factor = 1.0 - (*hr - 0.5).abs(); // Optimal around 50% of range

        (rest_factor * 0.4 + activity_factor * 0.4 + hr_factor * 0.2).clamp(0.0, 1.0)
    }
}

impl Default for HealthLearner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================
// Location Learning (CoreLocation/MapKit)
// ============================================

/// Location category for privacy-preserving learning
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LocationCategory {
    /// Home location
    Home = 0,
    /// Work location
    Work = 1,
    /// Gym/fitness
    Gym = 2,
    /// Restaurant/dining
    Dining = 3,
    /// Shopping/retail
    Shopping = 4,
    /// Entertainment venue
    Entertainment = 5,
    /// Outdoor/park
    Outdoor = 6,
    /// Transit (commuting)
    Transit = 7,
    /// Medical/healthcare
    Healthcare = 8,
    /// Social gathering
    Social = 9,
    /// Unknown/other
    Unknown = 255,
}

/// Privacy-preserving location state
#[derive(Clone, Debug)]
pub struct LocationState {
    /// Current location category (not actual coordinates)
    pub category: LocationCategory,
    /// Time at current location (minutes)
    pub duration_minutes: u32,
    /// Movement speed category (0=stationary, 1=walking, 2=driving)
    pub movement_type: u8,
    /// Hour of day
    pub hour: u8,
    /// Day of week
    pub day_of_week: u8,
    /// Is commuting (between home/work)
    pub is_commuting: bool,
}

impl LocationState {
    /// Convert to feature vector
    pub fn to_features(&self) -> Vec<f32> {
        let mut features = vec![0.0; 16];

        // One-hot encode category (0-9)
        let cat = self.category as usize;
        if cat < 10 {
            features[cat] = 1.0;
        }

        // Duration normalized (10)
        features[10] = (self.duration_minutes as f32 / 180.0).min(1.0);

        // Movement type (11)
        features[11] = self.movement_type as f32 / 2.0;

        // Time encoding (12-14)
        features[12] = (self.hour as f32 * std::f32::consts::PI / 12.0).sin();
        features[13] = (self.hour as f32 * std::f32::consts::PI / 12.0).cos();
        features[14] = self.day_of_week as f32 / 6.0;

        // Commuting flag (15)
        features[15] = if self.is_commuting { 1.0 } else { 0.0 };

        features
    }
}

/// Location pattern learner
pub struct LocationLearner {
    /// Transition probabilities: from_category -> to_category -> probability
    transitions: Vec<Vec<f32>>,
    /// Time spent at each category by hour
    time_by_hour: Vec<Vec<f32>>,
    /// Visit counts
    visit_counts: Vec<u32>,
    /// Total transitions
    total_transitions: u64,
}

impl LocationLearner {
    pub fn new() -> Self {
        Self {
            transitions: vec![vec![0.0; 10]; 10],
            time_by_hour: vec![vec![0.0; 10]; 24],
            visit_counts: vec![0; 10],
            total_transitions: 0,
        }
    }

    /// Learn from location transition
    pub fn learn_transition(&mut self, from: LocationCategory, to: LocationCategory) {
        let from_idx = (from as usize).min(9);
        let to_idx = (to as usize).min(9);

        // Increment transition count
        self.transitions[from_idx][to_idx] += 1.0;
        self.visit_counts[to_idx] += 1;
        self.total_transitions += 1;

        // Normalize row
        let row_sum: f32 = self.transitions[from_idx].iter().sum();
        if row_sum > 0.0 {
            for j in 0..10 {
                self.transitions[from_idx][j] /= row_sum;
            }
        }
    }

    /// Learn time spent at location
    pub fn learn_time(&mut self, state: &LocationState) {
        let cat = (state.category as usize).min(9);
        let hour = (state.hour as usize) % 24;

        // Exponential moving average
        let alpha = 0.1;
        self.time_by_hour[hour][cat] =
            (1.0 - alpha) * self.time_by_hour[hour][cat] + alpha * (state.duration_minutes as f32 / 60.0);
    }

    /// Predict next likely location
    pub fn predict_next(&self, current: LocationCategory) -> Vec<(LocationCategory, f32)> {
        let from_idx = (current as usize).min(9);
        let mut predictions: Vec<(LocationCategory, f32)> = (0..10)
            .filter_map(|i| {
                let prob = self.transitions[from_idx][i];
                if prob > 0.05 {
                    Some((unsafe { std::mem::transmute(i as u8) }, prob))
                } else {
                    None
                }
            })
            .collect();

        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        predictions
    }

    /// Get typical location for hour
    pub fn typical_location(&self, hour: u8) -> LocationCategory {
        let h = (hour as usize) % 24;
        let mut max_idx = 0;
        let mut max_val = 0.0;

        for i in 0..10 {
            if self.time_by_hour[h][i] > max_val {
                max_val = self.time_by_hour[h][i];
                max_idx = i;
            }
        }

        unsafe { std::mem::transmute(max_idx as u8) }
    }
}

impl Default for LocationLearner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================
// Communication Pattern Learning
// ============================================

/// Communication event type (metadata only, no content)
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum CommEventType {
    /// Incoming call
    IncomingCall = 0,
    /// Outgoing call
    OutgoingCall = 1,
    /// Missed call
    MissedCall = 2,
    /// Incoming message
    IncomingMessage = 3,
    /// Outgoing message
    OutgoingMessage = 4,
}

/// Privacy-preserving communication pattern
#[derive(Clone, Debug)]
pub struct CommPattern {
    /// Events per hour (24 slots)
    pub hourly_events: Vec<u32>,
    /// Average response time (seconds, 0 = N/A)
    pub avg_response_time: f32,
    /// Preferred communication hours (bit flags for 24 hours)
    pub preferred_hours: u32,
    /// Do-not-disturb score (0-1, higher = less likely to respond)
    pub dnd_score: f32,
}

impl Default for CommPattern {
    fn default() -> Self {
        Self {
            hourly_events: vec![0; 24],
            avg_response_time: 300.0, // 5 minutes default
            preferred_hours: 0x00FFFE00, // 9am-11pm default
            dnd_score: 0.0,
        }
    }
}

/// Communication learner
pub struct CommLearner {
    /// Event counts by hour
    event_counts: Vec<Vec<u32>>,
    /// Response times (moving average)
    response_times: Vec<f32>,
    /// Total events
    total_events: u64,
}

impl CommLearner {
    pub fn new() -> Self {
        Self {
            event_counts: vec![vec![0; 5]; 24], // 24 hours x 5 event types
            response_times: vec![300.0; 24],    // Default 5 min response time
            total_events: 0,
        }
    }

    /// Learn from communication event
    pub fn learn_event(&mut self, event_type: CommEventType, hour: u8, response_time_secs: Option<f32>) {
        let h = (hour as usize) % 24;
        let e = event_type as usize;

        self.event_counts[h][e] += 1;
        self.total_events += 1;

        if let Some(rt) = response_time_secs {
            let alpha = 0.1;
            self.response_times[h] = (1.0 - alpha) * self.response_times[h] + alpha * rt;
        }
    }

    /// Get communication pattern
    pub fn get_pattern(&self) -> CommPattern {
        let mut pattern = CommPattern::default();

        // Sum events per hour
        for h in 0..24 {
            pattern.hourly_events[h] = self.event_counts[h].iter().sum();
        }

        // Calculate preferred hours (above median activity)
        let median = {
            let mut sorted: Vec<u32> = pattern.hourly_events.clone();
            sorted.sort();
            sorted[12]
        };

        pattern.preferred_hours = 0;
        for h in 0..24 {
            if pattern.hourly_events[h] > median {
                pattern.preferred_hours |= 1 << h;
            }
        }

        // Average response time
        pattern.avg_response_time = self.response_times.iter().sum::<f32>() / 24.0;

        pattern
    }

    /// Check if current hour is good for communication
    pub fn is_good_time(&self, hour: u8) -> f32 {
        let h = (hour as usize) % 24;
        let total: u32 = self.event_counts[h].iter().sum();
        let max_total: u32 = self.event_counts.iter().map(|v| v.iter().sum::<u32>()).max().unwrap_or(1);

        if max_total == 0 {
            return 0.5;
        }

        total as f32 / max_total as f32
    }
}

impl Default for CommLearner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================
// Context Fusion & Master Learner
// ============================================

/// Combined iOS context for holistic learning
#[derive(Clone, Debug, Default)]
pub struct iOSContext {
    /// Health state
    pub health: Option<HealthState>,
    /// Location state
    pub location: Option<LocationState>,
    /// Hour of day
    pub hour: u8,
    /// Day of week
    pub day_of_week: u8,
    /// Is device locked
    pub device_locked: bool,
    /// Battery level (0-1)
    pub battery_level: f32,
    /// Network type (0=none, 1=wifi, 2=cellular)
    pub network_type: u8,
}

impl iOSContext {
    /// Convert to unified feature vector
    pub fn to_features(&self) -> Vec<f32> {
        let mut features = Vec::new();

        // Health features (20 dims)
        if let Some(ref health) = self.health {
            features.extend(health.to_features());
        } else {
            features.extend(vec![0.0; 20]);
        }

        // Location features (16 dims)
        if let Some(ref location) = self.location {
            features.extend(location.to_features());
        } else {
            features.extend(vec![0.0; 16]);
        }

        // Device state (4 dims)
        features.push(if self.device_locked { 0.0 } else { 1.0 });
        features.push(self.battery_level);
        features.push(self.network_type as f32 / 2.0);
        features.push(0.0); // Reserved

        // Time (already in health/location, but add global)
        features.push((self.hour as f32 * std::f32::consts::PI / 12.0).sin());
        features.push((self.hour as f32 * std::f32::consts::PI / 12.0).cos());
        features.push(self.day_of_week as f32 / 6.0);
        features.push(0.0); // Reserved

        features // Total: 44 dims
    }
}

/// Master iOS learner combining all signals
pub struct iOSLearner {
    /// Health learner
    pub health: HealthLearner,
    /// Location learner
    pub location: LocationLearner,
    /// Communication learner
    pub comm: CommLearner,
    /// Context embeddings (learned patterns)
    context_embeddings: Vec<Vec<f32>>,
    /// Preference weights (learned from feedback)
    preference_weights: Vec<f32>,
    /// Total learning iterations
    iterations: u64,
}

impl iOSLearner {
    pub fn new() -> Self {
        Self {
            health: HealthLearner::new(),
            location: LocationLearner::new(),
            comm: CommLearner::new(),
            context_embeddings: Vec::new(),
            preference_weights: vec![1.0; 44], // Match feature dimensions
            iterations: 0,
        }
    }

    /// Learn from iOS context
    pub fn learn(&mut self, context: &iOSContext) {
        // Learn from individual components
        if let Some(ref health) = context.health {
            self.health.learn(health);
        }

        if let Some(ref location) = context.location {
            self.location.learn_time(location);
        }

        // Store context embedding for pattern matching
        let features = context.to_features();
        if self.context_embeddings.len() >= 1000 {
            self.context_embeddings.remove(0);
        }
        self.context_embeddings.push(features);

        self.iterations += 1;
    }

    /// Learn from user feedback (positive/negative reward)
    pub fn learn_from_feedback(&mut self, context: &iOSContext, reward: f32) {
        let features = context.to_features();

        // Update preference weights based on reward
        let learning_rate = 0.01;
        for (i, &f) in features.iter().enumerate() {
            if i < self.preference_weights.len() {
                // If feature was active and reward was positive, increase weight
                self.preference_weights[i] += learning_rate * reward * f;
                // Clamp to reasonable range
                self.preference_weights[i] = self.preference_weights[i].clamp(0.1, 10.0);
            }
        }
    }

    /// Get context score (how good is this context for user)
    pub fn score_context(&self, context: &iOSContext) -> f32 {
        let features = context.to_features();

        let mut score = 0.0;
        for (i, &f) in features.iter().enumerate() {
            if i < self.preference_weights.len() {
                score += f * self.preference_weights[i];
            }
        }

        // Normalize to 0-1
        (score / self.preference_weights.len() as f32).clamp(0.0, 1.0)
    }

    /// Find similar past contexts
    pub fn find_similar_contexts(&self, context: &iOSContext, k: usize) -> Vec<(usize, f32)> {
        let query = context.to_features();

        let mut similarities: Vec<(usize, f32)> = self
            .context_embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| {
                let sim = cosine_similarity(&query, emb);
                (i, sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);
        similarities
    }

    /// Get personalized recommendations based on context
    pub fn get_recommendations(&self, context: &iOSContext) -> ContextRecommendations {
        let mut recs = ContextRecommendations::default();

        // Energy-based recommendations
        if let Some(ref health) = context.health {
            let energy = self.health.estimate_energy(health);
            recs.suggested_activity = if energy > 0.7 {
                ActivitySuggestion::HighEnergy
            } else if energy > 0.4 {
                ActivitySuggestion::Moderate
            } else {
                ActivitySuggestion::Rest
            };
            recs.energy_level = energy;
        }

        // Communication timing
        recs.good_time_to_communicate = self.comm.is_good_time(context.hour);

        // Location-based suggestions
        if let Some(ref location) = context.location {
            let predictions = self.location.predict_next(location.category);
            if let Some((next, prob)) = predictions.first() {
                if *prob > 0.3 {
                    recs.predicted_next_location = Some(*next);
                }
            }
        }

        // Focus time detection
        recs.is_focus_time = self.detect_focus_time(context);

        // Overall context score
        recs.context_quality = self.score_context(context);

        recs
    }

    /// Detect if user is likely in focus/work mode
    fn detect_focus_time(&self, context: &iOSContext) -> bool {
        // Work hours (9-17) + at work location + low communication
        let is_work_hour = context.hour >= 9 && context.hour <= 17;
        let at_work = context
            .location
            .as_ref()
            .map(|l| l.category == LocationCategory::Work)
            .unwrap_or(false);
        let low_comm = self.comm.is_good_time(context.hour) < 0.3;

        is_work_hour && (at_work || low_comm)
    }

    /// Serialize learner state
    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Magic + version
        bytes.extend_from_slice(b"IOSL");
        bytes.extend_from_slice(&1u32.to_le_bytes());

        // Iterations
        bytes.extend_from_slice(&self.iterations.to_le_bytes());

        // Preference weights
        bytes.extend_from_slice(&(self.preference_weights.len() as u32).to_le_bytes());
        for &w in &self.preference_weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }

        // Note: Full serialization would include all sub-learners
        // Simplified for initial implementation

        bytes
    }

    /// Deserialize learner state
    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 || &bytes[0..4] != b"IOSL" {
            return None;
        }

        let _version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let iterations = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15],
        ]);

        let mut offset = 16;
        if offset + 4 > bytes.len() {
            return None;
        }

        let weights_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3],
        ]) as usize;
        offset += 4;

        let mut preference_weights = Vec::with_capacity(weights_len);
        for _ in 0..weights_len {
            if offset + 4 > bytes.len() {
                return None;
            }
            let w = f32::from_le_bytes([
                bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3],
            ]);
            preference_weights.push(w);
            offset += 4;
        }

        let mut learner = Self::new();
        learner.iterations = iterations;
        learner.preference_weights = preference_weights;
        Some(learner)
    }
}

impl Default for iOSLearner {
    fn default() -> Self {
        Self::new()
    }
}

/// Activity suggestion based on context
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActivitySuggestion {
    HighEnergy,
    Moderate,
    Rest,
    Focus,
    Social,
}

impl Default for ActivitySuggestion {
    fn default() -> Self {
        ActivitySuggestion::Moderate
    }
}

/// Context-based recommendations
#[derive(Clone, Debug, Default)]
pub struct ContextRecommendations {
    /// Suggested activity type
    pub suggested_activity: ActivitySuggestion,
    /// Energy level (0-1)
    pub energy_level: f32,
    /// Good time to send messages/calls (0-1)
    pub good_time_to_communicate: f32,
    /// Predicted next location
    pub predicted_next_location: Option<LocationCategory>,
    /// Is user likely in focus mode
    pub is_focus_time: bool,
    /// Overall context quality score (0-1)
    pub context_quality: f32,
}

// Helper function
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

// ============================================
// WASM Exports
// ============================================

static mut IOS_LEARNER: Option<iOSLearner> = None;

/// Initialize iOS learner
#[no_mangle]
pub extern "C" fn ios_learner_init() -> i32 {
    unsafe {
        IOS_LEARNER = Some(iOSLearner::new());
    }
    0
}

/// Learn from health data
#[no_mangle]
pub extern "C" fn ios_learn_health(
    steps: f32,
    active_energy: f32,
    heart_rate: f32,
    sleep_hours: f32,
    hour: u8,
    day_of_week: u8,
) {
    unsafe {
        if let Some(learner) = IOS_LEARNER.as_mut() {
            let health = HealthState::from_healthkit(
                steps,
                active_energy,
                heart_rate,
                sleep_hours,
                hour,
                day_of_week,
            );
            learner.health.learn(&health);
        }
    }
}

/// Learn from location
#[no_mangle]
pub extern "C" fn ios_learn_location(
    category: u8,
    duration_minutes: u32,
    movement_type: u8,
    hour: u8,
    day_of_week: u8,
) {
    unsafe {
        if let Some(learner) = IOS_LEARNER.as_mut() {
            let location = LocationState {
                category: if category < 10 {
                    unsafe { std::mem::transmute(category) }
                } else {
                    LocationCategory::Unknown
                },
                duration_minutes,
                movement_type,
                hour,
                day_of_week,
                is_commuting: false,
            };
            learner.location.learn_time(&location);
        }
    }
}

/// Learn from communication event
#[no_mangle]
pub extern "C" fn ios_learn_comm(event_type: u8, hour: u8, response_time_secs: f32) {
    unsafe {
        if let Some(learner) = IOS_LEARNER.as_mut() {
            let evt = if event_type < 5 {
                unsafe { std::mem::transmute(event_type) }
            } else {
                CommEventType::IncomingMessage
            };
            let rt = if response_time_secs > 0.0 {
                Some(response_time_secs)
            } else {
                None
            };
            learner.comm.learn_event(evt, hour, rt);
        }
    }
}

/// Get energy estimate
#[no_mangle]
pub extern "C" fn ios_get_energy(
    steps: f32,
    active_energy: f32,
    heart_rate: f32,
    sleep_hours: f32,
    hour: u8,
    day_of_week: u8,
) -> f32 {
    unsafe {
        if let Some(learner) = IOS_LEARNER.as_ref() {
            let health = HealthState::from_healthkit(
                steps,
                active_energy,
                heart_rate,
                sleep_hours,
                hour,
                day_of_week,
            );
            learner.health.estimate_energy(&health)
        } else {
            0.5
        }
    }
}

/// Check if good time to communicate
#[no_mangle]
pub extern "C" fn ios_is_good_comm_time(hour: u8) -> f32 {
    unsafe {
        if let Some(learner) = IOS_LEARNER.as_ref() {
            learner.comm.is_good_time(hour)
        } else {
            0.5
        }
    }
}

/// Get learner iterations
#[no_mangle]
pub extern "C" fn ios_learner_iterations() -> u64 {
    unsafe { IOS_LEARNER.as_ref().map(|l| l.iterations).unwrap_or(0) }
}

// ============================================
// Browser Bindings (wasm-bindgen)
// ============================================

#[cfg(feature = "browser")]
pub mod browser {
    use super::*;
    use wasm_bindgen::prelude::*;
    use serde::Serialize;

    /// iOS Learner for browser - JavaScript-friendly API
    #[wasm_bindgen]
    pub struct IOSLearnerJS {
        learner: iOSLearner,
    }

    #[wasm_bindgen]
    impl IOSLearnerJS {
        /// Create a new iOS learner
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            Self {
                learner: iOSLearner::new(),
            }
        }

        /// Learn from health data
        #[wasm_bindgen(js_name = "learnHealth")]
        pub fn learn_health(
            &mut self,
            steps: f32,
            active_energy: f32,
            heart_rate: f32,
            sleep_hours: f32,
            hour: u8,
            day_of_week: u8,
        ) {
            let health = HealthState::from_healthkit(
                steps,
                active_energy,
                heart_rate,
                sleep_hours,
                hour,
                day_of_week,
            );
            self.learner.health.learn(&health);
        }

        /// Learn from location data
        #[wasm_bindgen(js_name = "learnLocation")]
        pub fn learn_location(
            &mut self,
            category: u8,
            duration_minutes: u32,
            movement_type: u8,
            hour: u8,
            day_of_week: u8,
        ) {
            let cat = category_from_u8(category);
            let location = LocationState {
                category: cat,
                duration_minutes,
                movement_type,
                hour,
                day_of_week,
                is_commuting: false,
            };
            self.learner.location.learn_time(&location);
        }

        /// Learn from communication event
        #[wasm_bindgen(js_name = "learnComm")]
        pub fn learn_comm(&mut self, event_type: u8, hour: u8, response_time_secs: Option<f32>) {
            let evt = event_from_u8(event_type);
            self.learner.comm.learn_event(evt, hour, response_time_secs);
        }

        /// Get energy estimate for current health state
        #[wasm_bindgen(js_name = "getEnergy")]
        pub fn get_energy(
            &self,
            steps: f32,
            active_energy: f32,
            heart_rate: f32,
            sleep_hours: f32,
            hour: u8,
            day_of_week: u8,
        ) -> f32 {
            let health = HealthState::from_healthkit(
                steps,
                active_energy,
                heart_rate,
                sleep_hours,
                hour,
                day_of_week,
            );
            self.learner.health.estimate_energy(&health)
        }

        /// Check if good time to communicate
        #[wasm_bindgen(js_name = "isGoodCommTime")]
        pub fn is_good_comm_time(&self, hour: u8) -> f32 {
            self.learner.comm.is_good_time(hour)
        }

        /// Get total learning iterations
        #[wasm_bindgen(getter)]
        pub fn iterations(&self) -> u64 {
            self.learner.iterations
        }

        /// Predict next location from current location
        #[wasm_bindgen(js_name = "predictNextLocation")]
        pub fn predict_next_location(&self, current_category: u8) -> JsValue {
            let cat = category_from_u8(current_category);
            let predictions = self.learner.location.predict_next(cat);
            let result: Vec<(u8, f32)> = predictions.iter()
                .map(|(c, p)| (*c as u8, *p))
                .collect();
            serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
        }

        /// Get recommendations based on current context
        #[wasm_bindgen(js_name = "getRecommendations")]
        pub fn get_recommendations(
            &self,
            steps: f32,
            active_energy: f32,
            heart_rate: f32,
            sleep_hours: f32,
            hour: u8,
            day_of_week: u8,
            location_category: u8,
            duration_minutes: u32,
        ) -> JsValue {
            let health = HealthState::from_healthkit(
                steps, active_energy, heart_rate, sleep_hours, hour, day_of_week,
            );
            let cat = category_from_u8(location_category);
            let location = LocationState {
                category: cat,
                duration_minutes,
                movement_type: 0,
                hour,
                day_of_week,
                is_commuting: false,
            };
            let context = iOSContext {
                health: Some(health),
                location: Some(location),
                hour,
                day_of_week,
                device_locked: false,
                battery_level: 1.0,
                network_type: 1, // WiFi
            };
            let recs = self.learner.get_recommendations(&context);

            #[derive(Serialize)]
            struct RecsJS {
                energy_level: f32,
                suggested_activity: String,
                good_time_to_communicate: f32,
                is_focus_time: bool,
                context_quality: f32,
            }

            let js_recs = RecsJS {
                energy_level: recs.energy_level,
                suggested_activity: format!("{:?}", recs.suggested_activity),
                good_time_to_communicate: recs.good_time_to_communicate,
                is_focus_time: recs.is_focus_time,
                context_quality: recs.context_quality,
            };

            serde_wasm_bindgen::to_value(&js_recs).unwrap_or(JsValue::NULL)
        }

        /// Serialize learner state to bytes
        #[wasm_bindgen(js_name = "serialize")]
        pub fn serialize(&self) -> Vec<u8> {
            self.learner.serialize()
        }

        /// Deserialize learner from bytes
        #[wasm_bindgen(js_name = "deserialize")]
        pub fn deserialize(bytes: &[u8]) -> Option<IOSLearnerJS> {
            iOSLearner::deserialize(bytes).map(|learner| IOSLearnerJS { learner })
        }
    }

    impl Default for IOSLearnerJS {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Health Learner for browser - simpler API
    #[wasm_bindgen]
    pub struct HealthLearnerJS {
        learner: HealthLearner,
        sample_count: u32,
    }

    #[wasm_bindgen]
    impl HealthLearnerJS {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            Self {
                learner: HealthLearner::new(),
                sample_count: 0,
            }
        }

        /// Learn from health data
        #[wasm_bindgen(js_name = "learn")]
        pub fn learn(
            &mut self,
            steps: f32,
            active_energy: f32,
            heart_rate: f32,
            sleep_hours: f32,
            hour: u8,
            day_of_week: u8,
        ) {
            let state = HealthState::from_healthkit(
                steps, active_energy, heart_rate, sleep_hours, hour, day_of_week,
            );
            self.learner.learn(&state);
            self.sample_count += 1;
        }

        /// Estimate energy level
        #[wasm_bindgen(js_name = "estimateEnergy")]
        pub fn estimate_energy(
            &self,
            steps: f32,
            active_energy: f32,
            heart_rate: f32,
            sleep_hours: f32,
            hour: u8,
            day_of_week: u8,
        ) -> f32 {
            let state = HealthState::from_healthkit(
                steps, active_energy, heart_rate, sleep_hours, hour, day_of_week,
            );
            self.learner.estimate_energy(&state)
        }

        #[wasm_bindgen(getter)]
        pub fn iterations(&self) -> u32 {
            self.sample_count
        }
    }

    impl Default for HealthLearnerJS {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Location Learner for browser
    #[wasm_bindgen]
    pub struct LocationLearnerJS {
        learner: LocationLearner,
    }

    #[wasm_bindgen]
    impl LocationLearnerJS {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            Self {
                learner: LocationLearner::new(),
            }
        }

        /// Learn a location transition
        #[wasm_bindgen(js_name = "learnTransition")]
        pub fn learn_transition(&mut self, from_category: u8, to_category: u8) {
            let from = category_from_u8(from_category);
            let to = category_from_u8(to_category);
            self.learner.learn_transition(from, to);
        }

        /// Predict next location (returns array of [category, probability])
        #[wasm_bindgen(js_name = "predictNext")]
        pub fn predict_next(&self, current: u8) -> JsValue {
            let cat = category_from_u8(current);
            let predictions = self.learner.predict_next(cat);
            let result: Vec<(u8, f32)> = predictions.iter()
                .map(|(c, p)| (*c as u8, *p))
                .collect();
            serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
        }
    }

    impl Default for LocationLearnerJS {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Communication Learner for browser
    #[wasm_bindgen]
    pub struct CommLearnerJS {
        learner: CommLearner,
    }

    #[wasm_bindgen]
    impl CommLearnerJS {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            Self {
                learner: CommLearner::new(),
            }
        }

        /// Learn from a communication event
        #[wasm_bindgen(js_name = "learnEvent")]
        pub fn learn_event(&mut self, event_type: u8, hour: u8, response_time_secs: Option<f32>) {
            let evt = event_from_u8(event_type);
            self.learner.learn_event(evt, hour, response_time_secs);
        }

        /// Check if it's a good time to communicate
        #[wasm_bindgen(js_name = "isGoodTime")]
        pub fn is_good_time(&self, hour: u8) -> f32 {
            self.learner.is_good_time(hour)
        }
    }

    impl Default for CommLearnerJS {
        fn default() -> Self {
            Self::new()
        }
    }

    // Helper function for location category conversion
    fn category_from_u8(val: u8) -> LocationCategory {
        match val {
            0 => LocationCategory::Home,
            1 => LocationCategory::Work,
            2 => LocationCategory::Gym,
            3 => LocationCategory::Dining,
            4 => LocationCategory::Shopping,
            5 => LocationCategory::Entertainment,
            6 => LocationCategory::Outdoor,
            7 => LocationCategory::Transit,
            8 => LocationCategory::Healthcare,
            9 => LocationCategory::Social,
            _ => LocationCategory::Unknown,
        }
    }

    // Helper function for comm event type conversion
    fn event_from_u8(val: u8) -> CommEventType {
        match val {
            0 => CommEventType::IncomingCall,
            1 => CommEventType::OutgoingCall,
            2 => CommEventType::MissedCall,
            3 => CommEventType::IncomingMessage,
            _ => CommEventType::OutgoingMessage,
        }
    }

    /// Location category constants for JavaScript
    #[wasm_bindgen]
    pub struct LocationCategories;

    #[wasm_bindgen]
    impl LocationCategories {
        #[wasm_bindgen(getter)]
        pub fn home() -> u8 { 0 }
        #[wasm_bindgen(getter)]
        pub fn work() -> u8 { 1 }
        #[wasm_bindgen(getter)]
        pub fn gym() -> u8 { 2 }
        #[wasm_bindgen(getter)]
        pub fn dining() -> u8 { 3 }
        #[wasm_bindgen(getter)]
        pub fn shopping() -> u8 { 4 }
        #[wasm_bindgen(getter)]
        pub fn entertainment() -> u8 { 5 }
        #[wasm_bindgen(getter)]
        pub fn outdoor() -> u8 { 6 }
        #[wasm_bindgen(getter)]
        pub fn transit() -> u8 { 7 }
        #[wasm_bindgen(getter)]
        pub fn healthcare() -> u8 { 8 }
        #[wasm_bindgen(getter)]
        pub fn social() -> u8 { 9 }
        #[wasm_bindgen(getter)]
        pub fn unknown() -> u8 { 10 }
    }

    /// Communication event type constants for JavaScript
    #[wasm_bindgen]
    pub struct CommEventTypes;

    #[wasm_bindgen]
    impl CommEventTypes {
        #[wasm_bindgen(getter)]
        pub fn incoming_call() -> u8 { 0 }
        #[wasm_bindgen(getter)]
        pub fn outgoing_call() -> u8 { 1 }
        #[wasm_bindgen(getter)]
        pub fn missed_call() -> u8 { 2 }
        #[wasm_bindgen(getter)]
        pub fn incoming_message() -> u8 { 3 }
        #[wasm_bindgen(getter)]
        pub fn outgoing_message() -> u8 { 4 }
    }
}

// Re-export browser module when feature is enabled
#[cfg(feature = "browser")]
pub use browser::*;

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_learner() {
        let mut learner = HealthLearner::new();

        // Learn some patterns
        for hour in 0..24 {
            let state = HealthState::from_healthkit(
                hour as f32 * 500.0, // Steps increase during day
                hour as f32 * 30.0,  // Active energy
                70.0 + (hour as f32 % 12.0) * 2.0, // Heart rate varies
                7.5,
                hour,
                1, // Monday
            );
            learner.learn(&state);
        }

        // Check energy estimation
        let morning = HealthState::from_healthkit(2000.0, 100.0, 72.0, 7.5, 8, 1);
        let energy = learner.estimate_energy(&morning);
        assert!(energy > 0.0 && energy <= 1.0);
    }

    #[test]
    fn test_location_learner() {
        let mut learner = LocationLearner::new();

        // Learn home -> work transition
        learner.learn_transition(LocationCategory::Home, LocationCategory::Transit);
        learner.learn_transition(LocationCategory::Transit, LocationCategory::Work);
        learner.learn_transition(LocationCategory::Home, LocationCategory::Work);
        learner.learn_transition(LocationCategory::Home, LocationCategory::Work);

        // Predict next from home
        let predictions = learner.predict_next(LocationCategory::Home);
        assert!(!predictions.is_empty());
        // Work should be most likely
        assert_eq!(predictions[0].0, LocationCategory::Work);
    }

    #[test]
    fn test_ios_context() {
        let context = iOSContext {
            health: Some(HealthState::from_healthkit(5000.0, 200.0, 75.0, 7.0, 10, 2)),
            location: Some(LocationState {
                category: LocationCategory::Work,
                duration_minutes: 120,
                movement_type: 0,
                hour: 10,
                day_of_week: 2,
                is_commuting: false,
            }),
            hour: 10,
            day_of_week: 2,
            device_locked: false,
            battery_level: 0.8,
            network_type: 1,
        };

        let features = context.to_features();
        assert_eq!(features.len(), 44);
    }

    #[test]
    fn test_ios_learner() {
        let mut learner = iOSLearner::new();

        let context = iOSContext {
            health: Some(HealthState::from_healthkit(5000.0, 200.0, 75.0, 7.0, 10, 2)),
            location: None,
            hour: 10,
            day_of_week: 2,
            device_locked: false,
            battery_level: 0.8,
            network_type: 1,
        };

        learner.learn(&context);
        assert_eq!(learner.iterations, 1);

        let recs = learner.get_recommendations(&context);
        assert!(recs.energy_level >= 0.0 && recs.energy_level <= 1.0);
    }
}
