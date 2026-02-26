export interface SourceInfo {
  id: string;
  title: string;
  score: number;
  risk_level: string;
  text?: string;
  source?: string;
  doc_type?: string;
  collection?: string;
}

export interface AnalyzeRequest {
  clause_text: string;
  strategy: string;
  top_k: number;
}

export interface KeyIssue {
  issue: string;
  severity: 'high' | 'medium' | 'low';
  citation_indices?: number[];
}

export interface ClauseAnalysis {
  risk_level: string;
  assumptions: string[];
  key_issues: (KeyIssue | string)[];
  suggested_revisions: string;
  jurisdiction_notes?: string;
  confidence: string;
  confidence_rationale: string;
}

export interface AnalyzeResponse {
  analysis: ClauseAnalysis | string;
  sources: SourceInfo[];
  strategy: string;
  model: string;
  top_k: number;
  review_status: string;
  disclaimer: string;
}

export interface ContractReviewRequest {
  contract_text: string;
  playbook: string;
}

export interface ClauseReviewDetail {
  clause_type: string;
  risk_level: string;
  playbook_match: 'preferred' | 'fallback' | 'walk_away' | 'not_covered';
  extracted_text: string;
  preferred_position?: string;
  gaps: { issue: string; severity: string; playbook_says?: string; clause_says?: string }[];
  suggested_redline?: string;
  negotiation_notes?: string;
}

export interface ContractReviewSummary {
  total_clauses_reviewed: number;
  preferred_match: number;
  fallback_match: number;
  walk_away_triggered: number;
  not_in_playbook: number;
  overall_risk: string;
  critical_issues: string[];
}

export interface ContractReviewResponse {
  playbook: string;
  total_clauses: number;
  summary: ContractReviewSummary;
  clause_analyses: ClauseReviewDetail[];
  review_status: string;
  disclaimer: string;
}

export interface BreachRequest {
  data_types_compromised: string[];
  affected_states: string[];
  number_of_affected_individuals: number | string;
  encryption_status: string;
  entity_type: string;
  date_of_discovery: string | null;
}

export interface BreachSummary {
  total_jurisdictions: number;
  notifications_required: number;
  ag_notifications_required: string[];
  earliest_deadline: string;
  earliest_deadline_state?: string;
  safe_harbor_applies?: boolean;
  safe_harbor_reason?: string;
}

export interface BreachStateAnalysis {
  jurisdiction: string;
  notification_required: boolean | null;
  rationale: string;
  deadline: string;
  notify_ag: boolean | null;
  ag_notification_details: string;
  safe_harbor_applies: boolean | null;
  special_considerations: string[];
  confidence: string;
}

export interface BreachResponse {
  breach_params: Record<string, unknown>;
  summary: BreachSummary;
  state_analyses: BreachStateAnalysis[];
  review_status: string;
  disclaimer: string;
}

export interface KBSearchRequest {
  query: string;
  top_k: number;
  use_router: boolean;
}

export interface KBAnswer {
  answer: string;
  caveats?: string[];
  related_queries?: string[];
}

export interface KBSearchResponse {
  answer: KBAnswer | string;
  routing: Record<string, unknown>;
  sources: SourceInfo[];
  review_status: string;
  disclaimer: string;
}

export interface HealthResponse {
  status: string;
  provider: string;
  vector_store: string;
  document_count: number;
  available_strategies: string[];
}
