import { Component, input, signal, ElementRef, viewChildren } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { SourceInfo } from '../../models/api.models';

@Component({
  selector: 'app-sources-panel',
  imports: [DecimalPipe],
  templateUrl: './sources-panel.html',
  styleUrl: './sources-panel.scss',
})
export class SourcesPanelComponent {
  sources = input<SourceInfo[]>([]);
  provenance = input<string>('');
  confidence = input<string>('');
  confidenceReason = input<string>('');

  expandedIndex = signal<number | null>(null);
  highlightedIndex = signal<number | null>(null);
  sourceCards = viewChildren<ElementRef>('sourceCard');

  collectionLabel(source: SourceInfo): string {
    const s = source.source || source.collection || '';
    const map: Record<string, string> = {
      clauses_json: 'Curated Clauses',
      cuad: 'CUAD',
      statutes: 'Statute',
      common_paper: 'Playbook',
    };
    return map[s] || s;
  }

  toggleExpand(index: number): void {
    this.expandedIndex.set(this.expandedIndex() === index ? null : index);
  }

  scrollToSource(index: number): void {
    this.expandedIndex.set(index);
    this.highlightedIndex.set(index);

    const cards = this.sourceCards();
    if (cards[index]) {
      cards[index].nativeElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    setTimeout(() => this.highlightedIndex.set(null), 2200);
  }
}
