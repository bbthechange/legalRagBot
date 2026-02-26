import { Component, signal, inject, OnInit } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { ApiService } from './services/api.service';
import { HealthResponse } from './models/api.models';
import { ClauseAnalysisComponent } from './components/clause-analysis/clause-analysis';
import { ContractReviewComponent } from './components/contract-review/contract-review';
import { BreachResponseComponent } from './components/breach-response/breach-response';
import { KnowledgeBaseComponent } from './components/knowledge-base/knowledge-base';

@Component({
  selector: 'app-root',
  imports: [
    DecimalPipe,
    ClauseAnalysisComponent,
    ContractReviewComponent,
    BreachResponseComponent,
    KnowledgeBaseComponent,
  ],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App implements OnInit {
  private readonly api = inject(ApiService);

  activeTab = signal<'clause' | 'contract' | 'breach' | 'kb'>('clause');
  healthStatus = signal<string>('connecting');
  healthDetails = signal<HealthResponse | null>(null);

  tabs = [
    { id: 'clause' as const, label: 'Clause Analysis' },
    { id: 'contract' as const, label: 'Contract Review' },
    { id: 'breach' as const, label: 'Breach Response' },
    { id: 'kb' as const, label: 'Knowledge Base' },
  ];

  ngOnInit(): void {
    this.api.health().subscribe({
      next: (res) => {
        this.healthStatus.set(res.status);
        this.healthDetails.set(res);
      },
      error: () => {
        this.healthStatus.set('degraded');
      },
    });
  }
}
