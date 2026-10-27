(() => {
    const bootstrap = JSON.parse(document.getElementById('bootstrap-data').textContent);
    let state = bootstrap.state;
    let previousState = null;
    let preview = null;
    let chart = null;
    let page = 1;
    let pageSize = 10;
    let busy = false;
    const element = id => document.getElementById(id);
    const copy = value => JSON.parse(JSON.stringify(value));

    function setAssistantOpen(open) {
        element('assistant-panel').hidden = !open;
        element('assistant-toggle').setAttribute('aria-expanded', String(open));
        element('assistant-toggle').setAttribute('aria-label', open ? 'Close assistant' : 'Open assistant');
        (open ? element('question') : element('assistant-toggle')).focus({preventScroll: true});
    }

    element('assistant-toggle').addEventListener('click', () => setAssistantOpen(element('assistant-panel').hidden));
    element('assistant-close').addEventListener('click', () => setAssistantOpen(false));
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && !element('assistant-panel').hidden) {
            event.preventDefault();
            setAssistantOpen(false);
        }
    });

    function setBusy(value) {
        busy = value;
        element('busy').hidden = !value;
        document.querySelectorAll('#assistant-form button, #filter-form button, [data-question], #reset-filters, #previous-page, #next-page, #page-size').forEach(control => control.disabled = value);
        element('undo-view').disabled = value || !previousState;
        if (!value && preview) renderPreview();
    }

    function showError(error) {
        element('assistant-error').textContent = error.message;
        element('assistant-error').hidden = false;
    }

    async function post(path, body) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 50000);
        try {
            const response = await fetch(path, {method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body), signal: controller.signal});
            const result = await response.json();
            if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Please check the filter values.');
            return result;
        } catch (error) {
            if (error.name === 'AbortError') throw new Error('The request timed out. Filters were not changed.');
            throw error;
        } finally {
            clearTimeout(timeout);
        }
    }

    function drawChart(data) {
        if (chart) chart.destroy();
        if (typeof Chart === 'undefined') {
            element('chart-error').hidden = false;
            return;
        }
        chart = new Chart(element('revenue-chart'), {
            type: 'line',
            data: {labels: data.dates, datasets: [{label: 'Revenue · USD', data: data.revenue_usd, borderColor: '#206bc4', backgroundColor: 'rgba(32,107,196,.08)', borderWidth: 2, pointRadius: 0, fill: true, tension: .15}]},
            options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {x: {ticks: {maxTicksLimit: 8}}, y: {beginAtZero: true}}},
        });
    }

    function activateTab() {
        document.querySelectorAll('#dashboard-content [data-bs-toggle="tab"]').forEach(button => {
            const active = button.dataset.bsTarget === '#' + state.tab;
            button.classList.toggle('active', active);
            button.setAttribute('aria-selected', String(active));
        });
        document.querySelectorAll('#dashboard-content .tab-pane').forEach(pane => {
            const active = pane.id === state.tab;
            pane.classList.toggle('active', active);
            pane.classList.toggle('show', active);
        });
        if (chart) chart.resize();
    }

    function syncControls() {
        const filters = state.filters || {};
        element('email-prefixes').value = (filters.email_prefixes || []).join(', ');
        element('person-name').value = filters.person_name || 'any';
        element('title-contains').value = filters.title_contains || '';
        element('date-from').value = filters.start_date || '';
        element('date-to').value = filters.end_date || '';
        const labels = [];
        if (filters.email_prefixes?.length) labels.push('email: ' + filters.email_prefixes.join(' / '));
        if (filters.person_name && filters.person_name !== 'any') labels.push('PERSON: ' + filters.person_name);
        if (filters.title_contains) labels.push('title: ' + filters.title_contains);
        if (filters.author_contains) labels.push('author: ' + filters.author_contains);
        if (filters.genres?.length) labels.push('genre: ' + filters.genres.join(', '));
        if (filters.start_date || filters.end_date) labels.push((filters.start_date || '…') + ' — ' + (filters.end_date || '…'));
        element('view-scope').textContent = `STORE${state.store_id} · ` + (labels.join(' · ') || 'All records');
        element('undo-view').disabled = !previousState || busy;
    }

    function renderPreview() {
        if (!preview) return;
        const users = state.tab !== 'books';
        element('preview-title').textContent = users ? 'All users' : 'All books';
        const table = users ? preview.users : preview.books;
        const headers = users ? ['User ID', 'Canonical ID', 'Email (masked)'] : ['Book ID', 'Title', 'Authors', 'PERSON'];
        const head = element('preview-head');
        const body = element('preview-body');
        head.replaceChildren(); body.replaceChildren();
        const heading = document.createElement('tr');
        headers.forEach(title => {const cell = document.createElement('th'); cell.textContent = title; heading.append(cell);});
        head.append(heading);
        table.rows.forEach(row => {
            const values = users ? [row.user_id, row.canonical_user_id, row.email] : [row.book_id, row.title, row.authors.join(', '), row.person_names === null ? 'Index unavailable' : row.person_names.join(', ') || '—'];
            const tr = document.createElement('tr');
            values.forEach(value => {const td = document.createElement('td'); td.textContent = value; tr.append(td);});
            body.append(tr);
        });
        const first = table.total ? (page - 1) * pageSize + 1 : 0;
        const last = Math.min(page * pageSize, table.total);
        const pages = Math.max(1, Math.ceil(table.total / pageSize));
        element('preview-count').textContent = `Showing ${first}–${last} of ${table.total.toLocaleString('en-US')} records · Page ${page} of ${pages}` + (users ? ` · ${preview.users.unique.toLocaleString('en-US')} unique users` : '');
        element('previous-page').disabled = busy || page <= 1;
        element('next-page').disabled = busy || page * pageSize >= table.total;
    }

    function commitView(result, nextState, remember=true) {
        if (remember) previousState = copy(state);
        state = copy(nextState);
        preview = result.preview;
        page = preview.page;
        pageSize = preview.page_size;
        element('page-size').value = String(pageSize);
        if (chart) {chart.destroy(); chart = null;}
        element('dashboard-content').innerHTML = result.dashboard_html;
        document.title = `STORE${state.store_id} | Bookstore Analytics`;
        history.replaceState(null, '', `/store/${state.store_id}`);
        drawChart(result.dashboard.chart);
        activateTab(); syncControls(); renderPreview();
        try {sessionStorage.setItem('bookstore-view', JSON.stringify(state));} catch {}
    }

    async function changeView(nextState, nextPage=1, remember=true, nextPageSize=pageSize) {
        if (busy) return;
        setBusy(true); element('assistant-error').hidden = true;
        try {
            const result = await post('/api/view', {state: nextState, page: nextPage, page_size: nextPageSize});
            commitView(result, result.preview.state, remember);
        } catch (error) {
            showError(error);
            element('page-size').value = String(pageSize);
        } finally {setBusy(false);}
    }

    element('assistant-form').addEventListener('submit', async event => {
        event.preventDefault();
        if (busy || !element('question').value.trim()) return;
        setBusy(true); element('assistant-error').hidden = true;
        element('assistant-answer').textContent = '';
        element('evidence-box').hidden = true;
        try {
            const result = await post('/api/assistant', {message: element('question').value.trim(), state, page_size: pageSize});
            if (result.action && result.view) commitView(result.view, result.action.state);
            element('assistant-answer').textContent = result.answer;
            element('evidence').textContent = JSON.stringify(result.results, null, 2);
            element('evidence-box').hidden = !result.results.length;
        } catch (error) {showError(error);} finally {setBusy(false);}
    });

    document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => {element('question').value = button.dataset.question; element('question').focus();}));
    element('filter-form').addEventListener('submit', event => {
        event.preventDefault();
        const filters = {...state.filters, email_prefixes: element('email-prefixes').value.split(',').map(s => s.trim()).filter(Boolean), person_name: element('person-name').value, title_contains: element('title-contains').value, start_date: element('date-from').value || null, end_date: element('date-to').value || null};
        const tab = filters.person_name !== 'any' || filters.title_contains ? 'books' : filters.email_prefixes.length ? 'users' : state.tab;
        changeView({...state, filters, tab});
    });
    element('reset-filters').addEventListener('click', () => changeView({...state, filters: {}}));
    element('undo-view').addEventListener('click', () => {if (previousState) changeView(copy(previousState));});
    element('previous-page').addEventListener('click', () => changeView(state, page - 1, false));
    element('next-page').addEventListener('click', () => changeView(state, page + 1, false));
    element('page-size').addEventListener('change', event => changeView(state, 1, false, Number(event.target.value)));
    document.addEventListener('click', event => {
        const tab = event.target.closest('#dashboard-content [data-bs-toggle="tab"]');
        if (tab) {
            event.preventDefault(); event.stopPropagation();
            if (busy) return;
            changeView({...state, tab: tab.dataset.bsTarget.slice(1)}, 1, false);
        }
        const link = event.target.closest('#dashboard-content a[href^="/store/"]');
        if (link && !event.ctrlKey && !event.metaKey) {
            event.preventDefault();
            changeView({store_id: Number(link.getAttribute('href').split('/').pop()), tab: 'revenue', filters: {}});
        }
    }, true);

    drawChart(bootstrap.chart);
    syncControls();
    let saved = null;
    try {saved = JSON.parse(sessionStorage.getItem('bookstore-view'));} catch {}
    const next = saved?.store_id === state.store_id ? saved : state;
    changeView(next, 1, false);
})();
