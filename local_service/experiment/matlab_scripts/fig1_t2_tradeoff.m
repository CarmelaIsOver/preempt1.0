%% fig1_t2_tradeoff.m
% T2 数值脱敏隐私-效用权衡：ε vs MRE (DAG约束传播 vs 独立扰动)
% 数据源: experiment_results_v2.json (116样本, 3轮平均)
% 图表类型: 双面板散点连线图

clear; close all;

% ===== 读取数据 =====
data = readtable('../../figure_data/fig1_t2_tradeoff.dat', ...
    'FileType', 'text', 'CommentStyle', '#');

epsilon   = data.epsilon;
dag_mre   = data.dag_mre_pct;
nodag_mre = data.nodag_mre_pct;
dag_con   = data.dag_constraint_pct;
nodag_con = data.nodag_constraint_pct;

% ===== 颜色定义 (学术配色) =====
c_dag   = [0.17 0.48 0.71];   % 蓝 — DAG
c_nodag = [0.84 0.10 0.11];   % 红 — No-DAG
c_fill  = [0.84 0.10 0.11];   % 填充色

% ===== 创建图窗 =====
figure('Position', [100 100 1100 450], 'Color', 'w');

% ---- 左面板: MRE 对比 ----
subplot(1,2,1);
h1 = plot(epsilon, dag_mre, 'o-', 'Color', c_dag, 'LineWidth', 2.5, ...
    'MarkerSize', 10, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', c_dag);
hold on;
h2 = plot(epsilon, nodag_mre, 's--', 'Color', c_nodag, 'LineWidth', 2.5, ...
    'MarkerSize', 10, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', c_nodag);

% 填充 MRE 差距区域
x_fill = [epsilon; flipud(epsilon)];
y_fill = [dag_mre; flipud(nodag_mre)];
fill(x_fill, y_fill, c_fill, 'FaceAlpha', 0.08, 'EdgeColor', 'none');

set(gca, 'XScale', 'log', 'XMinorGrid', 'off');
xlabel('\epsilon (隐私预算)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('均值相对误差 MRE (%)', 'FontSize', 13, 'FontWeight', 'bold');
title('T2 数值扰动精度对比', 'FontSize', 14, 'FontWeight', 'bold');
legend([h1 h2], {'DAG 约束传播 (Anon)', '独立扰动 (No-DAG)'}, ...
    'Location', 'northeast', 'FontSize', 10, 'Box', 'off');
grid on; set(gca, 'GridAlpha', 0.15);
xlim([0.08 12]);
ylim([-0.5 12]);

% 标注推荐配置
text(1.0, dag_mre(epsilon==1.0)+1.2, ...
    ['\leftarrow \epsilon=1.0, MRE=' num2str(dag_mre(epsilon==1.0), '%.2f') '%'], ...
    'Color', c_dag, 'FontSize', 10, 'FontWeight', 'bold');

% ---- 右面板: 约束满足率 ----
subplot(1,2,2);
h3 = plot(epsilon, dag_con, 'o-', 'Color', c_dag, 'LineWidth', 2.5, ...
    'MarkerSize', 10, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', c_dag);
hold on;
h4 = plot(epsilon, nodag_con, 's--', 'Color', c_nodag, 'LineWidth', 2.5, ...
    'MarkerSize', 10, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', c_nodag);

x_fill2 = [epsilon; flipud(epsilon)];
y_fill2 = [dag_con; flipud(nodag_con)];
fill(x_fill2, y_fill2, c_fill, 'FaceAlpha', 0.08, 'EdgeColor', 'none');

set(gca, 'XScale', 'log');
xlabel('\epsilon (隐私预算)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('约束满足率 (%)', 'FontSize', 13, 'FontWeight', 'bold');
title('DAG 约束满足率对比', 'FontSize', 14, 'FontWeight', 'bold');
legend([h3 h4], {'DAG 约束传播 (恒99.6%)', '独立扰动 (崩溃至50.9%)'}, ...
    'Location', 'southeast', 'FontSize', 10, 'Box', 'off');
grid on; set(gca, 'GridAlpha', 0.15);
ylim([0 108]);

% 标注
text(2.0, 72, ['\epsilon=0.1时约束' newline '满足率崩溃至50.9%'], ...
    'Color', c_nodag, 'FontSize', 9, 'FontStyle', 'italic');

% ===== 全局标题 =====
sgtitle('RQ2：T2 数值脱敏隐私-效用权衡 (\epsilon-LDP)', ...
    'FontSize', 15, 'FontWeight', 'bold');

% ===== 保存 =====
exportgraphics(gcf, 'fig1_t2_tradeoff.pdf', 'ContentType', 'vector', 'Resolution', 300);
fprintf('✅ fig1_t2_tradeoff.pdf 已保存\n');
