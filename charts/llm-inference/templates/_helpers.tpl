{{- define "llm.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "llm.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
llm.platform/engine: {{ .Values.engine }}
{{- end -}}

{{- define "llm.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/* Prometheus metric prefix differs per engine */}}
{{- define "llm.metricPrefix" -}}
{{- if eq .Values.engine "vllm" -}}vllm:{{- else -}}{{- end -}}
{{- end -}}

{{- define "llm.validate" -}}
{{- if not (has .Values.engine (list "nim" "vllm")) -}}
{{- fail "engine must be either 'nim' or 'vllm'" -}}
{{- end -}}
{{- if and (gt (int .Values.replicaCount) 1) .Values.modelCache.enabled (ne .Values.modelCache.accessMode "ReadWriteMany") -}}
{{- fail "more than one replica sharing the model cache needs accessMode ReadWriteMany" -}}
{{- end -}}
{{- end -}}
