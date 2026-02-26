import { Component, inject, signal, computed, ViewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { KBSearchResponse, KBAnswer, SourceInfo } from '../../models/api.models';
import { SourcesPanelComponent } from '../sources-panel/sources-panel';

@Component({
  selector: 'app-knowledge-base',
  imports: [FormsModule, SourcesPanelComponent],
  templateUrl: './knowledge-base.html',
  styleUrl: './knowledge-base.scss',
})
export class KnowledgeBaseComponent {
  private readonly api = inject(ApiService);

  @ViewChild(SourcesPanelComponent) sourcesPanel!: SourcesPanelComponent;

  query = '';
  loading = signal(false);
  error = signal<string | null>(null);
  response = signal<KBSearchResponse | null>(null);
  inputExpanded = signal(true);

  answer = computed<KBAnswer | null>(() => {
    const r = this.response();
    if (!r) return null;
    if (typeof r.answer === 'string') {
      return { answer: r.answer };
    }
    return r.answer;
  });

  sources = computed<SourceInfo[]>(() => this.response()?.sources || []);

  routing = computed<string>(() => {
    const r = this.response();
    if (!r?.routing) return '';
    const rt = r.routing as any;
    const parts: string[] = [];
    if (rt.query_type) parts.push(rt.query_type);
    if (rt.search_strategy) parts.push(rt.search_strategy);
    if (rt.filters) {
      const fStr = Object.entries(rt.filters)
        .filter(([, v]) => v)
        .map(([k, v]) => `${k}=${v}`)
        .join(', ');
      if (fStr) parts.push(fStr);
    }
    return parts.length ? `Searched: ${parts.join(' / ')}` : '';
  });

  relatedQuestions = computed<string[]>(() => {
    const a = this.answer();
    return a?.related_queries || [];
  });

  search(): void {
    if (!this.query.trim() || this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    this.response.set(null);

    this.api.searchKnowledgeBase({
      query: this.query,
      top_k: 5,
      use_router: true,
    }).subscribe({
      next: (res) => {
        this.response.set(res);
        this.loading.set(false);
        this.inputExpanded.set(false);
      },
      error: (err) => {
        this.error.set(err?.error?.detail || err?.message || 'Search failed. Check that the backend is running.');
        this.loading.set(false);
      },
    });
  }

  searchRelated(question: string): void {
    this.query = question;
    this.search();
  }

  onCitationClick(index: number): void {
    this.sourcesPanel?.scrollToSource(index);
  }
}
