const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Header, Footer, AlignmentType,
  HeadingLevel, PageNumber, PageBreak, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, LevelFormat, TableOfContents,
} = require('docx');

const LOGO = fs.readFileSync(process.argv[2]);
const OUT = process.argv[3];
const FONT = 'Times New Roman';
const MAROON = '8B0000';

const logo = (w, h) => new ImageRun({ type: 'png', data: LOGO, transformation: { width: w, height: h } });

const p = (text, opts = {}) => new Paragraph({
  alignment: opts.align ?? AlignmentType.JUSTIFIED,
  spacing: { after: 160, line: 360 },
  children: [new TextRun({ text, italics: opts.italic, bold: opts.bold, color: opts.color })],
});
const todo = (text) => p(`[TO DO: ${text}]`, { italic: true, color: '7F7F7F' });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (t) => new Paragraph({ numbering: { reference: 'b', level: 0 }, spacing: { after: 80, line: 340 }, children: [new TextRun(t)] });
const caption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 240 }, children: [new TextRun({ text: t, italics: true, size: 20 })] });
const centered = (t, o = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: o.after ?? 120 }, children: [new TextRun({ text: t, bold: o.bold, size: o.size, color: o.color })] });

const border = { style: BorderStyle.SINGLE, size: 4, color: '808080' };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: 'F2DCDB', type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: t, bold: head, size: 20 })] })],
  });
  return new Table({
    alignment: AlignmentType.CENTER, width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
  });
}

const header = new Header({ children: [new Paragraph({
  tabStops: [{ type: 'right', position: 9026 }],
  border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: MAROON, space: 4 } },
  children: [logo(105, 35), new TextRun({ children: ['\t' + 'Deep Learning ESE Project Report 2026-27'], size: 18, color: '404040' })],
})] });

const footer = new Footer({ children: [new Paragraph({
  tabStops: [{ type: 'right', position: 9026 }],
  border: { top: { style: BorderStyle.SINGLE, size: 8, color: MAROON, space: 4 } },
  children: [logo(66, 22), new TextRun({ text: '   Symbiosis Institute of Technology, Pune', size: 18, color: '404040' }),
    new TextRun({ children: ['\tPage ', PageNumber.CURRENT, ' of ', PageNumber.TOTAL_PAGES], size: 18, color: '404040' })],
})] });

const TITLE = 'AI-Based Predictive Maintenance Using Multivariate Sensor Data';

const cover = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 400 }, children: [logo(270, 90)] }),
  centered('SYMBIOSIS INSTITUTE OF TECHNOLOGY, PUNE', { bold: true, size: 30, color: MAROON }),
  centered('Symbiosis International (Deemed University)', { size: 24, after: 600 }),
  centered('Deep Learning', { bold: true, size: 28 }),
  centered('ESE Project Report (CA3) | Academic Year 2026-27', { size: 24, after: 600 }),
  centered(TITLE, { bold: true, size: 36, after: 700 }),
  centered('Submitted by', { size: 24, after: 200 }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [table(
    ['Name', 'PRN', 'Branch', 'Division'],
    [['Saafalya Sengupta', '24070126158', 'AIML', 'C1'], ['Ribhav Dhankani', '24070126150', 'AIML', 'C1']],
    [3000, 2000, 1500, 1500])] }),
  centered('', { after: 400 }),
  centered('Under the guidance of: [Faculty Name]', { size: 24 }),
  centered('Department of Artificial Intelligence and Machine Learning', { size: 24 }),
  centered('Submission Date: [DD Month 2026]', { size: 24 }),
  new Paragraph({ children: [new PageBreak()] }),
];

const toc = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: 'Table of Contents', bold: true, size: 32 })] }),
  new TableOfContents('Table of Contents', { hyperlink: true, headingStyleRange: '1-2' }),
  p('(Right-click the table above and choose "Update Field" in Word to refresh page numbers.)', { italic: true, color: '7F7F7F' }),
  new Paragraph({ children: [new PageBreak()] }),
];

const body = [
  h1('1. Title'),
  p(TITLE, { bold: true }),
  p('Short title: Deep Learning-Based Remaining Useful Life (RUL) Prediction for Predictive Maintenance.'),

  h1('2. Abstract and Keywords'),
  p('Unplanned equipment failure causes costly downtime and safety risks in aviation, manufacturing and energy. Predictive maintenance (PdM) uses condition-monitoring data to estimate when a machine is likely to fail, so that maintenance can be scheduled just in time. This project builds a deep learning pipeline that predicts the Remaining Useful Life (RUL) of turbofan engines from multivariate time-series sensor data, using the NASA C-MAPSS benchmark dataset. Raw sensor streams are cleaned, normalised and converted into fixed-length sliding windows. We compare classical machine learning baselines (Linear Regression, Random Forest) with deep models: a 1D Convolutional Neural Network (CNN), a Long Short-Term Memory (LSTM) network, and a hybrid CNN-LSTM with attention. Models are evaluated with Root Mean Squared Error (RMSE) and the asymmetric PHM08 scoring function, which penalises late predictions more than early ones. [TO DO: add one sentence with the best model and its RMSE/score once experiments are done.] The results show that sequence-aware deep models capture degradation trends better than static baselines, supporting their use in real-world maintenance planning.'),
  p('Keywords: Predictive Maintenance, Remaining Useful Life, Multivariate Time Series, LSTM, 1D-CNN, Attention, C-MAPSS, Prognostics and Health Management.', { bold: true }),

  h1('3. Introduction'),
  h2('3.1 Background'),
  p('Industrial maintenance usually follows one of three strategies. Reactive maintenance repairs a machine after it breaks, which leads to long downtime. Preventive maintenance services machines at fixed intervals, which is safer but wastes the useful life of healthy parts. Predictive maintenance monitors the actual health of the machine through sensors and acts only when degradation is detected. With cheap IoT sensors and Industry 4.0 systems, large volumes of temperature, pressure, vibration and speed readings are now available, making data-driven PdM practical.'),
  p('A central task in PdM is estimating Remaining Useful Life (RUL): the number of operating cycles a component can still run before failure. RUL estimation from multivariate sensor data is hard because sensors are noisy, correlated with each other, affected by changing operating conditions, and degradation is gradual and non-linear. Deep learning models can learn these temporal patterns directly from raw sequences, reducing the need for hand-crafted features.'),
  h2('3.2 Problem Statement'),
  p('Given multivariate sensor readings recorded over the operating life of an engine, predict its RUL at any point in time, accurately and with a bias towards early (safe) rather than late (dangerous) predictions.'),
  h2('3.3 Objectives'),
  bullet('Pre-process the C-MAPSS dataset: select informative sensors, normalise readings and generate sliding-window sequences.'),
  bullet('Build and train deep learning models (1D-CNN, LSTM, CNN-LSTM with attention) for RUL regression.'),
  bullet('Compare the deep models against classical ML baselines using RMSE and the PHM08 score.'),
  bullet('Interpret results with prediction plots and error analysis to judge practical usefulness.'),
  h2('3.4 Scope and Report Organisation'),
  p('The study uses simulated turbofan engine data and focuses on RUL regression. Section 4 reviews related work, Section 5 describes the dataset and models, Section 6 presents results, Section 7 concludes and Section 8 lists references.'),

  h1('4. Literature Review'),
  p('Early prognostics work relied on physics-based degradation models and statistical methods such as similarity-based matching and regression on hand-crafted health indicators. Lei et al. [1] review the full PdM pipeline, from data acquisition and health-indicator construction to RUL prediction, and note that data-driven methods are increasingly preferred when accurate physical models are unavailable. Carvalho et al. [2] survey machine learning for PdM and find Random Forests, SVMs and neural networks to be the most used techniques.'),
  p('NASA released the C-MAPSS dataset [3], generated by simulating damage propagation in turbofan engines, and it has become the standard benchmark for RUL prediction. Ramasso and Saxena [4] analysed the methods applied to it and highlighted the importance of consistent evaluation metrics such as RMSE and the PHM08 score.'),
  p('Neural networks were applied to this problem early on: Heimes [5] used recurrent neural networks in the 2008 PHM challenge. Babu et al. [6] were among the first to apply a deep CNN to RUL estimation, treating sensor windows as 2D inputs and outperforming shallow models. Li et al. [7] improved on this with a deeper 1D-CNN and time-window inputs, reporting strong results on all four C-MAPSS subsets.'),
  p('Because degradation is a temporal process, recurrent models are a natural fit. The LSTM [8] solves the vanishing-gradient problem of plain RNNs and can learn long-range dependencies. Zheng et al. [9] showed that stacked LSTM layers predict RUL more accurately than CNN and classical baselines on C-MAPSS. Malhotra et al. [10] used an LSTM encoder-decoder for unsupervised anomaly detection in multi-sensor data, showing that reconstruction error can act as a health indicator.'),
  p('Attention mechanisms [11] let a model weight the most informative time steps. Recent studies combine CNN feature extractors, LSTM sequence encoders and attention layers into hybrid models, reporting further accuracy gains and better interpretability.'),
  h2('4.1 Research Gap'),
  p('Many studies report results on a single model or subset, with differing pre-processing, which makes fair comparison hard. This project compares classical and deep models under one consistent pipeline (same sensors, windows, RUL cap and metrics) and adds qualitative error analysis.'),
  table(['Ref.', 'Method', 'Dataset', 'Key Contribution'], [
    ['[6]', 'Deep CNN', 'C-MAPSS', 'First CNN-based RUL regression'],
    ['[7]', '1D-CNN with time windows', 'C-MAPSS', 'Deeper CNN, strong accuracy on all subsets'],
    ['[9]', 'Stacked LSTM', 'C-MAPSS', 'Temporal modelling beats CNN and ML baselines'],
    ['[10]', 'LSTM encoder-decoder', 'Multi-sensor', 'Unsupervised health indicator from reconstruction error'],
    ['This work', 'CNN-LSTM + attention', 'C-MAPSS', 'Unified comparison of ML and DL models'],
  ], [1200, 2600, 1600, 3626]),
  caption('Table 1: Summary of related work'),

  h1('5. Methodology'),
  h2('5.1 Dataset Description'),
  p('We use the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset [3]. Each engine starts in a healthy state with unknown initial wear and degrades until failure. Each row is one operating cycle and contains the engine ID, cycle number, 3 operational settings and 21 sensor readings (temperatures, pressures, fan and core speeds, fuel flow ratios, etc.). The training set runs every engine to failure; the test set stops some time before failure, and true RUL values are provided separately.'),
  table(['Subset', 'Train Engines', 'Test Engines', 'Operating Conditions', 'Fault Modes'], [
    ['FD001', '100', '100', '1', '1 (HPC)'],
    ['FD002', '260', '259', '6', '1 (HPC)'],
    ['FD003', '100', '100', '1', '2 (HPC, Fan)'],
    ['FD004', '248', '248', '6', '2 (HPC, Fan)'],
  ], [1600, 1800, 1800, 2000, 1826]),
  caption('Table 2: C-MAPSS dataset subsets'),
  todo('state which subsets are used (e.g. FD001 first, then FD003/FD004) and add a sensor-trend plot of a sample engine.'),
  h2('5.2 Data Pre-processing'),
  bullet('Sensor selection: drop sensors with near-constant values (e.g. s1, s5, s10, s16, s18, s19 in FD001), keeping about 14 informative sensors.'),
  bullet('Normalisation: Min-Max scaling fitted on training data only (per operating condition for FD002/FD004).'),
  bullet('RUL labelling: RUL = max cycle - current cycle, capped at 125 cycles (piece-wise linear target), since early-life degradation is not observable.'),
  bullet('Sliding windows: sequences of length 30 cycles with stride 1, giving input shape (window, sensors).'),
  bullet('Split: 80/20 split of training engines (by engine ID) for validation.'),
  h2('5.3 Model Architectures'),
  p('Baselines: Linear Regression and Random Forest Regressor trained on flattened windows.'),
  p('1D-CNN: Conv1D (64 filters, kernel 5) -> Conv1D (64, kernel 3) -> Global Average Pooling -> Dense (64) -> Dropout -> Dense (1).'),
  p('LSTM: LSTM (64, return sequences) -> LSTM (32) -> Dropout (0.2) -> Dense (32) -> Dense (1).'),
  p('Proposed CNN-LSTM with Attention: Conv1D layers extract local features, an LSTM encodes temporal dependencies, and an attention layer weights the most informative time steps before a dense regression head.'),
  todo('insert block diagram of the full pipeline: Raw Sensor Data -> Pre-processing -> Sliding Windows -> Model -> RUL Output -> Maintenance Decision.'),
  caption('Figure 1: Proposed system block diagram'),
  h2('5.4 Training Setup'),
  table(['Parameter', 'Value'], [
    ['Framework', 'Python, TensorFlow/Keras (or PyTorch)'],
    ['Loss function', 'Mean Squared Error'],
    ['Optimiser', 'Adam [12], learning rate 0.001'],
    ['Batch size / Epochs', '256 / up to 100 with early stopping'],
    ['Window length / RUL cap', '30 cycles / 125 cycles'],
  ], [3500, 5526]),
  caption('Table 3: Hyperparameters (update after tuning)'),
  h2('5.5 Evaluation Metrics'),
  p('RMSE = sqrt( (1/N) * sum (y_pred - y_true)^2 ), which treats early and late errors equally.'),
  p('PHM08 Score: s = sum of (exp(-d/13) - 1) when d < 0, and (exp(d/10) - 1) when d >= 0, where d = y_pred - y_true. Late predictions (d > 0) are penalised more heavily, since they are more dangerous in practice. Lower is better for both metrics.'),

  h1('6. Results and Discussion'),
  h2('6.1 Quantitative Results'),
  table(['Model', 'RMSE (FD001)', 'Score (FD001)', 'RMSE (FD003)', 'Training Time'], [
    ['Linear Regression', '-', '-', '-', '-'],
    ['Random Forest', '-', '-', '-', '-'],
    ['1D-CNN', '-', '-', '-', '-'],
    ['LSTM', '-', '-', '-', '-'],
    ['CNN-LSTM + Attention', '-', '-', '-', '-'],
  ], [2600, 1600, 1600, 1600, 1626]),
  caption('Table 4: Comparative results on the test set'),
  todo('fill Table 4 after running experiments; add a short paragraph on which model wins and by how much.'),
  h2('6.2 Qualitative Results'),
  todo('add (a) predicted vs. true RUL plot for test engines, (b) RUL trajectory for 2-3 sample engines, (c) training/validation loss curves, (d) attention-weight heatmap.'),
  caption('Figure 2: Predicted vs. actual RUL'),
  h2('6.3 Discussion'),
  todo('interpret results: why sequence models beat baselines, effect of the RUL cap and window length, early vs. late prediction errors, failure cases, and practical implications for maintenance scheduling.'),

  h1('7. Conclusion'),
  p('This project presents a deep learning approach to predictive maintenance that estimates the Remaining Useful Life of turbofan engines from multivariate sensor data. A consistent pipeline of sensor selection, normalisation, RUL capping and sliding windows is used to compare classical machine learning baselines with 1D-CNN, LSTM and a hybrid CNN-LSTM with attention. [TO DO: state the best model and key numbers.] Future work includes testing on real industrial data, Transformer-based models, uncertainty estimation for safer decisions, and deployment as a real-time monitoring dashboard.'),

  h1('8. References'),
  ...[
    'Y. Lei, N. Li, L. Guo, N. Li, T. Yan, and J. Lin, "Machinery health prognostics: A systematic review from data acquisition to RUL prediction," Mechanical Systems and Signal Processing, vol. 104, pp. 799-834, 2018.',
    'T. P. Carvalho, F. A. Soares, R. Vita, R. P. Francisco, J. P. Basto, and S. G. Alcala, "A systematic literature review of machine learning methods applied to predictive maintenance," Computers & Industrial Engineering, vol. 137, 106024, 2019.',
    'A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure simulation," in Proc. Int. Conf. Prognostics and Health Management (PHM), 2008, pp. 1-9.',
    'E. Ramasso and A. Saxena, "Performance benchmarking and analysis of prognostic methods for CMAPSS datasets," Int. Journal of Prognostics and Health Management, vol. 5, no. 2, 2014.',
    'F. O. Heimes, "Recurrent neural networks for remaining useful life estimation," in Proc. Int. Conf. Prognostics and Health Management (PHM), 2008, pp. 1-6.',
    'G. S. Babu, P. Zhao, and X.-L. Li, "Deep convolutional neural network based regression approach for estimation of remaining useful life," in Proc. DASFAA, LNCS vol. 9642, Springer, 2016, pp. 214-228.',
    'X. Li, Q. Ding, and J.-Q. Sun, "Remaining useful life estimation in prognostics using deep convolution neural networks," Reliability Engineering & System Safety, vol. 172, pp. 1-11, 2018.',
    'S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.',
    'S. Zheng, K. Ristovski, A. Farahat, and C. Gupta, "Long short-term memory network for remaining useful life estimation," in Proc. IEEE Int. Conf. Prognostics and Health Management (ICPHM), 2017, pp. 88-95.',
    'P. Malhotra, A. Ramakrishnan, G. Anand, L. Vig, P. Agarwal, and G. Shroff, "LSTM-based encoder-decoder for multi-sensor anomaly detection," arXiv:1607.00148, 2016.',
    'A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 5998-6008.',
    'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. Int. Conf. Learning Representations (ICLR), 2015.',
  ].map((r, i) => new Paragraph({ spacing: { after: 120 }, indent: { left: 540, hanging: 540 }, alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun(`[${i + 1}]\t${r}`)] })),
];

const doc = new Document({
  creator: 'Saafalya Sengupta, Ribhav Dhankani',
  title: TITLE,
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 30, bold: true, color: MAROON, font: FONT }, paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: FONT }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: 'b', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440, header: 600, footer: 600 } } },
    headers: { default: header }, footers: { default: footer },
    children: [...cover, ...toc, ...body],
  }],
});

Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log('written', OUT); });
