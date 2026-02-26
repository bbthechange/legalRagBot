import { Component, input, output, signal } from '@angular/core';

@Component({
  selector: 'app-action-bar',
  templateUrl: './action-bar.html',
  styleUrl: './action-bar.scss',
})
export class ActionBarComponent {
  hasEdits = input(false);
  sendForReview = output<void>();
  exportWord = output<void>();
  exportPdf = output<void>();
  routeToSpecialist = output<string>();

  specialists = [
    'IP Attorney',
    'Privacy Counsel',
    'Senior Partner',
    'Employment Counsel',
  ];

  showRouteMenu = signal(false);

  onSendForReview(): void {
    this.sendForReview.emit();
  }

  onExportWord(): void {
    this.exportWord.emit();
  }

  onExportPdf(): void {
    this.exportPdf.emit();
  }

  onRouteToSpecialist(specialist: string): void {
    this.showRouteMenu.set(false);
    this.routeToSpecialist.emit(specialist);
  }
}
