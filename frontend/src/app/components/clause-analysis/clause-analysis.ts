import { Component, inject, signal, computed, ViewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { AnalyzeResponse, ClauseAnalysis, KeyIssue, SourceInfo } from '../../models/api.models';
import { SourcesPanelComponent } from '../sources-panel/sources-panel';
import { ActionBarComponent } from '../action-bar/action-bar';
import { RiskBadgeComponent } from '../risk-badge/risk-badge';

@Component({
  selector: 'app-clause-analysis',
  imports: [FormsModule, SourcesPanelComponent, ActionBarComponent, RiskBadgeComponent],
  templateUrl: './clause-analysis.html',
  styleUrl: './clause-analysis.scss',
})
export class ClauseAnalysisComponent {
  private readonly api = inject(ApiService);

  @ViewChild(SourcesPanelComponent) sourcesPanel!: SourcesPanelComponent;

  clauseText = '';
  strategy = signal<string>('few_shot');
  loading = signal(false);
  error = signal<string | null>(null);
  response = signal<AnalyzeResponse | null>(null);
  editedRevision = signal<string>('');
  originalRevision = signal<string>('');
  inputExpanded = signal(true);

  strategies = [
    { id: 'few_shot', label: 'Few-Shot', hint: 'Uses worked examples for highest accuracy' },
    { id: 'structured', label: 'Structured', hint: 'Detailed role and JSON schema guidelines' },
    { id: 'basic', label: 'Basic', hint: 'Minimal system prompt baseline' },
  ];

  strategyLabel = computed(() =>
    this.strategies.find(s => s.id === this.strategy())?.label ?? this.strategy()
  );

  analysis = computed<ClauseAnalysis | null>(() => {
    const r = this.response();
    if (!r) return null;
    return typeof r.analysis === 'string' ? null : r.analysis;
  });

  sources = computed<SourceInfo[]>(() => this.response()?.sources || []);

  /** Normalize key_issues: backend may return string[] or KeyIssue[]. */
  normalizedKeyIssues = computed<KeyIssue[]>(() => {
    const a = this.analysis();
    if (!a?.key_issues?.length) return [];
    return a.key_issues.map(item =>
      typeof item === 'string' ? { issue: item, severity: 'medium' as const } : item
    );
  });

  provenance = computed<string>(() => {
    const r = this.response();
    if (!r) return '';
    const now = new Date();
    const ts = now.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + ' ' +
      now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    return `${r.model} / ${r.strategy} / ${r.sources.length} sources / ${ts}`;
  });

  hasEdits = computed(() => this.editedRevision() !== this.originalRevision());

  analyze(): void {
    if (!this.clauseText.trim() || this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    this.response.set(null);

    this.api.analyzeClause({
      clause_text: this.clauseText,
      strategy: this.strategy(),
      top_k: 3,
    }).subscribe({
      next: (res) => {
        this.response.set(res);
        const a = typeof res.analysis === 'string' ? null : res.analysis;
        const rev = a?.suggested_revisions || '';
        this.originalRevision.set(rev);
        this.editedRevision.set(rev);
        this.loading.set(false);
        this.inputExpanded.set(false);
      },
      error: (err) => {
        this.error.set(err?.error?.detail || err?.message || 'Analysis failed. Check that the backend is running.');
        this.loading.set(false);
      },
    });
  }

  onCitationClick(index: number): void {
    this.sourcesPanel?.scrollToSource(index);
  }

  severityClass(severity: string): string {
    const s = severity?.toLowerCase();
    if (s === 'high') return 'severity-high';
    if (s === 'medium') return 'severity-medium';
    return 'severity-low';
  }

  resetRevision(): void {
    this.editedRevision.set(this.originalRevision());
  }
}
