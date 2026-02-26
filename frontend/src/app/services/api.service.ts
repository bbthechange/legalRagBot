import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  AnalyzeRequest, AnalyzeResponse,
  ContractReviewRequest, ContractReviewResponse,
  BreachRequest, BreachResponse,
  KBSearchRequest, KBSearchResponse,
  HealthResponse,
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = '/api';

  private headers(): HttpHeaders {
    return new HttpHeaders({ 'Content-Type': 'application/json' });
  }

  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`);
  }

  analyzeClause(req: AnalyzeRequest): Observable<AnalyzeResponse> {
    return this.http.post<AnalyzeResponse>(`${this.baseUrl}/analyze`, req, { headers: this.headers() });
  }

  reviewContract(req: ContractReviewRequest): Observable<ContractReviewResponse> {
    return this.http.post<ContractReviewResponse>(`${this.baseUrl}/contract-review`, req, { headers: this.headers() });
  }

  analyzeBreach(req: BreachRequest): Observable<BreachResponse> {
    return this.http.post<BreachResponse>(`${this.baseUrl}/breach-analysis`, req, { headers: this.headers() });
  }

  searchKnowledgeBase(req: KBSearchRequest): Observable<KBSearchResponse> {
    return this.http.post<KBSearchResponse>(`${this.baseUrl}/ask`, req, { headers: this.headers() });
  }
}
