require 'cairo'

local HISTORY_SECONDS = 30
local last_update = -1
local last_sample_time = 0
local gpu_state = {}

local function trim(value)
    return (value:gsub('^%s+', ''):gsub('%s+$', ''))
end

local function csv_fields(line)
    local fields = {}
    for field in string.gmatch(line, '([^,]+)') do
        fields[#fields + 1] = trim(field)
    end
    return fields
end

local function shortened_name(name)
    return name:gsub('NVIDIA GeForce ', '')
end

local function append_sample(state, now, utilization, vram_percent)
    local samples = state.samples
    if #samples > 0 and samples[#samples].time == now then
        samples[#samples].utilization = utilization
        samples[#samples].vram_percent = vram_percent
    else
        samples[#samples + 1] = {
            time = now,
            utilization = utilization,
            vram_percent = vram_percent,
        }
    end

    local cutoff = now - HISTORY_SECONDS
    while #samples > 0 and samples[1].time < cutoff do
        table.remove(samples, 1)
    end
end

local function poll_gpus()
    local update = tonumber(conky_parse('${updates}')) or 0
    if update == last_update then
        return
    end
    last_update = update

    local command = "LC_ALL=C nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null"
    local handle = io.popen(command)
    if handle == nil then
        return
    end

    local now = os.time()
    local observed = false
    for line in handle:lines() do
        local f = csv_fields(line)
        if #f >= 8 then
            local index = tonumber(f[1])
            local used = tonumber(f[4])
            local total = tonumber(f[5])
            if index ~= nil and used ~= nil and total ~= nil and total > 0 then
                local state = gpu_state[index] or { samples = {} }
                state.name = shortened_name(f[2])
                state.utilization = tonumber(f[3]) or 0
                state.memory_used = used
                state.memory_total = total
                state.temperature = tonumber(f[6]) or 0
                state.power_draw = tonumber(f[7]) or 0
                state.power_limit = tonumber(f[8]) or 0
                state.last_seen = now
                append_sample(state, now, state.utilization, 100 * used / total)
                gpu_state[index] = state
                observed = true
            end
        end
    end
    handle:close()
    if observed then
        last_sample_time = now
    end
end

local function rgb(hex)
    local value = tonumber(hex, 16)
    return math.floor(value / 0x10000) % 0x100 / 255,
           math.floor(value / 0x100) % 0x100 / 255,
           value % 0x100 / 255
end

local function set_colour(cr, hex, alpha)
    local r, g, b = rgb(hex)
    cairo_set_source_rgba(cr, r, g, b, alpha or 1)
end

local function draw_text(cr, x, y, text, size, colour, bold)
    cairo_select_font_face(
        cr,
        'DejaVu Sans Mono',
        CAIRO_FONT_SLANT_NORMAL,
        bold and CAIRO_FONT_WEIGHT_BOLD or CAIRO_FONT_WEIGHT_NORMAL
    )
    cairo_set_font_size(cr, size)
    set_colour(cr, colour, 1)
    cairo_move_to(cr, x, y)
    cairo_show_text(cr, text)
end

local function draw_graph(cr, x, y, width, height, samples, field, colour, label, current)
    set_colour(cr, '17212d', 0.96)
    cairo_rectangle(cr, x, y, width, height)
    cairo_fill(cr)

    cairo_set_line_width(cr, 1)
    set_colour(cr, '8294aa', 0.20)
    for fraction = 0.25, 0.75, 0.25 do
        local gy = y + height * (1 - fraction)
        cairo_move_to(cr, x, gy)
        cairo_line_to(cr, x + width, gy)
    end
    for seconds = 10, 20, 10 do
        local gx = x + width * seconds / HISTORY_SECONDS
        cairo_move_to(cr, gx, y)
        cairo_line_to(cr, gx, y + height)
    end
    cairo_stroke(cr)

    draw_text(cr, x + 14, y + 27, label, 20, 'd7e3f4', true)
    draw_text(cr, x + width - 90, y + 27, string.format('%3.0f%%', current or 0), 20, colour, true)

    if #samples == 0 then
        return
    end

    local now = os.time()
    local start_time = now - HISTORY_SECONDS
    local points = {}
    for _, sample in ipairs(samples) do
        if sample.time >= start_time then
            local px = x + width * (sample.time - start_time) / HISTORY_SECONDS
            local value = math.max(0, math.min(100, sample[field] or 0))
            local py = y + height - (height - 34) * value / 100
            points[#points + 1] = { x = px, y = py }
        end
    end

    if #points == 0 then
        return
    end

    if #points > 1 then
        set_colour(cr, colour, 0.17)
        cairo_move_to(cr, points[1].x, y + height)
        for _, point in ipairs(points) do
            cairo_line_to(cr, point.x, point.y)
        end
        cairo_line_to(cr, points[#points].x, y + height)
        cairo_close_path(cr)
        cairo_fill(cr)
    end

    cairo_set_line_width(cr, 2.5)
    cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND)
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND)
    set_colour(cr, colour, 1)
    cairo_move_to(cr, points[1].x, points[1].y)
    for i = 2, #points do
        cairo_line_to(cr, points[i].x, points[i].y)
    end
    cairo_stroke(cr)

    if #points == 1 then
        cairo_arc(cr, points[1].x, points[1].y, 2.5, 0, 2 * math.pi)
        cairo_fill(cr)
    end
end

local function draw_gpu_row(cr, index, y, window_width)
    local state = gpu_state[index]
    local name = state and state.name or ('GPU ' .. tostring(index))
    draw_text(cr, 30, y + 27, string.format('GPU %d — %s', index, name), 24, 'ffcc66', true)

    if state == nil then
        draw_text(cr, 30, y + 55, 'Waiting for NVIDIA telemetry...', 19, '8294aa', false)
        return
    end

    local details = string.format(
        'Load %3.0f%%   VRAM %.1f / %.1f GiB   Temp %2.0f C   Power %.0f / %.0f W',
        state.utilization,
        state.memory_used / 1024,
        state.memory_total / 1024,
        state.temperature,
        state.power_draw,
        state.power_limit
    )
    draw_text(cr, 30, y + 56, details, 19, 'd7e3f4', false)

    local vram_percent = 100 * state.memory_used / state.memory_total
    local padding = 30
    local gap = 30
    local graph_width = (window_width - 2 * padding - gap) / 2
    draw_graph(cr, padding, y + 70, graph_width, 220, state.samples, 'utilization', '65d1ff', 'GPU USAGE', state.utilization)
    draw_graph(cr, padding + graph_width + gap, y + 70, graph_width, 220, state.samples, 'vram_percent', 'ffcc66', 'VRAM', vram_percent)
end

function conky_draw_gpu_graphs()
    if conky_window == nil then
        return
    end

    poll_gpus()

    local surface = cairo_xlib_surface_create(
        conky_window.display,
        conky_window.drawable,
        conky_window.visual,
        conky_window.width,
        conky_window.height
    )
    local cr = cairo_create(surface)

    draw_text(cr, 30, 252, 'GPU HISTORY  —  LAST 30 SECONDS', 22, '79e6a7', true)
    draw_gpu_row(cr, 0, 270, conky_window.width)
    draw_gpu_row(cr, 1, 580, conky_window.width)

    if last_sample_time > 0 and os.time() - last_sample_time > 3 then
        draw_text(cr, conky_window.width - 260, 252, 'TELEMETRY STALE', 18, 'ff6b6b', true)
    end

    cairo_destroy(cr)
    cairo_surface_destroy(surface)
end
